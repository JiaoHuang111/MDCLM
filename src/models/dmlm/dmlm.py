# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
#
# DMLM: Diffusion Material Language Model
# 该文件基于原始 DPLM (DiffusionProteinLanguageModel) 完整架构改写，
# 保留中间 backbone 与生成/解码逻辑，只把 tokenizer 替换为 CrystaTokenizerWrapper，
# 并给出 embedding 对齐的钩子，用于从头训练（不加载 dplm 权重）。
#
# 注意：
# - 请确保 CrystaTokenizerWrapper 的实现存在并可 import（下方尝试了几种导入方式）。
# - 若 get_net(...) 返回的 net 中内置了 embedding，需要在初始化时根据 crysta vocab_size 重新初始化 embedding。
# - 本文件尽量与原 dplm.py 保持一致（函数名/接口/行为），以便与训练 pipeline 无缝衔接。

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer  # 保留以防 net_class 需要

# 导入项目内部的注册器与工具函数（与 dplm.py 保持一致）
from byprot.models import register_model
from byprot.models.utils import (
    LoRAConfig,
    NetConfig,
    get_net,
    get_net_class,
    sample_from_categorical,
    stochastic_sample_from_categorical,
    top_k_top_p_filtering,
    topk_masking,
)
from byprot import utils

log = utils.get_logger(__name__)

# 尝试导入 CrystaTokenizerWrapper：你需要把 tokenizer 放在工程里下列路径之一
# 推荐位置： src/dmlm/tokenizers/crysta_tokenizer.py 或 项目根 dmlm/tokenizers/crysta_tokenizer.py
try:
    # 先尝试工程内模块风格导入（如果你把 tokenizer 放在 dmlm/tokenizers 下）
    from dmlm.tokenizers.crysta_tokenizer import CrystaTokenizerWrapper
except Exception:
    try:
        # 再尝试按 byprot 包内导入（如果你放入 byprot.tokenizers）
        from byprot.tokenizers.crysta_tokenizer import CrystaTokenizerWrapper
    except Exception:
        # 如果都不能导入，给出友好提示并在运行时抛错
        CrystaTokenizerWrapper = None


# 默认配置 dataclass（与 DPLMConfig 保持字段一致，方便替换配置）
@dataclass
class DMLMConfig:
    # 扩散时间步数量（和 DPLM 一致）
    num_diffusion_timesteps: int = field(default=500)
    # LoRA 配置（如果使用 LoRA）
    lora: LoRAConfig = field(default=LoRAConfig())
    # 网络配置（get_net 会使用）
    net: NetConfig = field(default=NetConfig())
    # 是否启用 gradient checkpoint
    gradient_ckpt: bool = field(default=False)
    # 是否启用 rdm_couple 耦合训练
    rdm_couple: bool = field(default=False)


# 注册模型名为 "dmlm"，Hydra/registry 可以用这个名字实例化
@register_model("dmlm")
class DiffusionMaterialLanguageModel(nn.Module):
    """DMLM：基于 DPLM 架构的晶体语言模型。

    说明：
    - backbone (self.net) 由 get_net(self.cfg) 创建（与 DPLM 保持一致）。
    - tokenizer 使用 CrystaTokenizerWrapper（需要在项目中准备好）。
    - 如果从头训练，请确保 embedding 大小与 tokenizer.vocab_size 对齐：
        - 如果 net 内部已有 embedding（比如 net.embed_tokens 或 net.lm_head），
          代码中提供了 `maybe_resize_token_embeddings` 的调用位置用于重新初始化 embedding。
    """

    _default_cfg = DMLMConfig()  # 默认配置

    def __init__(self, cfg, net=None, from_dplm_weights=False, crysta_meta=None, num_diffusion_timesteps=None, **kwargs):
        """
        Args:
            cfg: OmegaConf / dict 配置，会与 _default_cfg 合并
            net: 如果外部传入 net 实例则使用，否则由 get_net(self.cfg) 创建
            from_dplm_weights: 如果 True 尝试加载 dplm 权重（默认 False，因你要从头训练）
            crysta_meta: 可选，来自 meta.pkl 的 dict，用于初始化 tokenizer（如果需要）
        """
        log.info(f'Function DiffusionMaterialLanguageModel.__init__() start.')
        self.num_diffusion_timesteps = num_diffusion_timesteps
        super().__init__()

        # 合并并保存配置（将用户 cfg 与默认 cfg 合并）
        self._update_cfg(cfg)
        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 1: creating net.')
        # -------- 1) 初始化 backbone/net ----------
        # 如果外部传入 net，就直接使用；否则根据 cfg 创建（与 DPLM 一致）
        self.net = get_net(self.cfg) if net is None else net

        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 2: creating tokenizer.')
        # -------- 2) 初始化并替换 tokenizer ----------
        # 如果 CrystaTokenizerWrapper 未能 import，上面 CrystaTokenizerWrapper 会为 None
        if CrystaTokenizerWrapper is None:
            raise ImportError(
                "CrystaTokenizerWrapper 未找到。请将 dmlm/tokenizers/crysta_tokenizer.py 放到项目中并保证可 import。"
            )
        else:
            log.info(f'Function DiffusionMaterialLanguageModel.__init__(): Import CrystaTokenizerWrapper success.')

        # 创建 crysta tokenizer 实例；如果 meta（meta.pkl）可用，优先传入
        try:
            # 尝试以 meta 初始化（如果你传入了 crysta_meta）
            if crysta_meta is not None:
                self.tokenizer = CrystaTokenizerWrapper(meta=crysta_meta)
            else:
                self.tokenizer = CrystaTokenizerWrapper()
        except TypeError:
            # 如果封装器不接受 meta 参数，退回到无参构造
            self.tokenizer = CrystaTokenizerWrapper()
        log.info(f'Function DiffusionMaterialLanguageModel.__init__(): Instancing CrystaTokenizerWrapper success.')

        # 将 tokenizer 绑定到 net 上（覆盖 net 的 tokenizer，以确保一致）
        # 这样 net 在生成 logits 时如果依赖 self.net.tokenizer 可以使用 CrystaTokenizerWrapper
        try:
            self.net.tokenizer = self.tokenizer
        except Exception:
            # 如果 net 没有 tokenizer 属性，则忽略
            pass

        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 3: special token id.')
        # -------- 3) 特殊 token id（同 DPLM） ----------
        # 这些 id 期望从 net 或 tokenizer 提供
        # 优先从 net 取（保持原有行为），如果 net 没提供则从 tokenizer 取
        self.mask_id = getattr(self.net, "mask_id", None) or getattr(self.tokenizer, "mask_token_id", None)
        self.pad_id = getattr(self.net, "pad_id", None) or getattr(self.tokenizer, "pad_token_id", None)
        self.bos_id = getattr(self.net, "bos_id", None) or getattr(self.tokenizer, "bos_token_id", None)
        self.eos_id = getattr(self.net, "eos_id", None) or getattr(self.tokenizer, "eos_token_id", None)
        #  self.x_id = getattr(self.net, "x_id", None)  # 有些实现会用 x_id 代表特殊占位

        # -------- 4) 如果需要，从 dplm 权重加载（用户要求从头训练时无需启用） ----------
        if from_dplm_weights:
            # 这里保留接口，但默认不使用。若启用，get_net_class 等代码会尝试加载对应权重。
            # 具体加载逻辑可参考原 dplm.py 的 from_pretrained 实现（不在此重复）。
            pass

        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 5: vocab size.')
        # -------- 5) 若 net 的 embedding 与 Crysta vocab 不一致 -> 重新初始化 embedding ----------
        # 很多 net 实现会包含一个 embedding 层，例如 attribute 名为 "embed_tokens" 或 "embeddings.weight"
        # 我们尝试尽可能发现并调整 embedding 的大小以匹配 tokenizer.vocab_size
        crysta_vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if crysta_vocab_size is not None:
            # 尝试几种常见 embedding 属性名
            # 1) Common HF style: net.get_input_embeddings() / net.resize_token_embeddings
            if hasattr(self.net, "resize_token_embeddings"):
                # 如果 net 支持 resize（如 transformers-based），调用它
                log.info(f'net 支持 resize.')
                try:
                    self.net.resize_token_embeddings(crysta_vocab_size)
                    log.info(f'Resize token embedding Done.')
                except Exception:
                    # 如果失败，不产生致命错误，只打印提示
                    log.error(f'resize_token_embeddings 失败，请手动检查 net 的 embedding 并调整为 crysta_vocab_size。')
            else:
                # 2) 直接查找 embed_tokens 或 embeddings
                log.info(f'net 不支持 resize.')
                if hasattr(self.net, "embed_tokens"):
                    old = self.net.embed_tokens
                    if getattr(old, "num_embeddings", None) != crysta_vocab_size:
                        # 直接替换为新的 nn.Embedding 并初始化
                        hidden = old.embedding_dim if hasattr(old, "embedding_dim") else old.weight.size(1)
                        new_emb = nn.Embedding(crysta_vocab_size, hidden)
                        # 使用与原来相同的初始化方法
                        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
                        self.net.embed_tokens = new_emb
                        log.info(f"已将 net.embed_tokens 重置为大小 {crysta_vocab_size} x {hidden}")
                elif hasattr(self.net, "embeddings") and hasattr(self.net.embeddings, "word_embeddings"):
                    # ESM 风格或其他可能使用 embeddings.word_embeddings
                    we = self.net.embeddings.word_embeddings
                    if getattr(we, "num_embeddings", None) != crysta_vocab_size:
                        hidden = we.embedding_dim if hasattr(we, "embedding_dim") else we.weight.size(1)
                        new_we = nn.Embedding(crysta_vocab_size, hidden)
                        nn.init.normal_(new_we.weight, mean=0.0, std=0.02)
                        self.net.embeddings.word_embeddings = new_we
                        log.info(f"已将 net.embeddings.word_embeddings 重置为大小 {crysta_vocab_size} x {hidden}")
                else:
                    # 未能识别 embedding 结构，给出提示，用户可手动调整 net 定义以匹配 vocab_size
                    log.warning("警告：未检测到 net 常见 embedding 属性"
                                "（resize_token_embeddings/embed_tokens/embeddings.word_embeddings），"
                                "请手动确保 embedding 大小与 tokenizer.vocab_size 对齐。")

        # -------- 6) 如果配置开启 gradient checkpoint，则启用 net 的 checkpointing（与 DPLM 一致） ----------
        if self.cfg.gradient_ckpt:
            if hasattr(self.net, "supports_gradient_checkpointing"):
                self.net.supports_gradient_checkpointing = True
                try:
                    # 一些模型 API 支持 gradient_checkpointing_enable()
                    self.net.gradient_checkpointing_enable()
                except Exception:
                    pass
        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Done.')


    # 与 DPLM 保持一致的 from_pretrained 接口（保留，便于未来加载预训练权重）
    @classmethod
    def from_pretrained(
        cls, net_name, cfg_override={}, net_override={}, from_huggingface=False
    ):
        """
        参考 DPLM 的 from_pretrained 实现：保留本接口以便将来加载 checkpoint。
        """
        from pathlib import Path
        from collections import OrderedDict
        import json
        import torch

        if not from_huggingface:
            # local checkpoint 加载（与原 dplm 相同的逻辑）
            from byprot.utils.config import load_yaml_config

            # cfg_path = Path(net_name).parents[1]
            # cfg_path = Path(cfg_path, ".hydra", "config.yaml")
            cfg_path = Path("configs", "config_all.yaml")
            # 加载完整配置
            full_cfg = load_yaml_config(str(cfg_path))
            print(f"Loaded config keys: {list(full_cfg.keys())}")  # 调试信息
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            # 安全地移除 _target_
            if "_target_" in cfg:
                cfg.pop("_target_")
            model = cls(cfg)

            pretrained_state_dict = torch.load(
                net_name, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()

            # remove the "model." prefix if present
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v

            missing, unexpected = model.load_state_dict(
                new_pretrained_state_dict, strict=False
            )
            print(
                f"Restored from {net_name} with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
                print(f"Unexpected Keys: {unexpected}")
            return model
        else:
            # 如果需要从 HuggingFace 或本地 HF mirror 加载网络（保留接口）
            # 这里示例使用 local_dir 方式（如 dplm.py 中），你可以按需修改
            local_dir = "airkingbd/dmlm_650m"  # 如果存在本地 HF 风格仓库可改为你的路径
            if local_dir is None:
                raise ValueError(
                    "`local_dir` must be provided when `from_huggingface=True` and server cannot access HuggingFace."
                )

            config_path = Path(local_dir, "config.json")
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")

            with open(config_path, "r") as f:
                config = json.load(f)
            dplm_type = config.get("dplm_type")  # 保持字段名以兼容原有实现（可能需要改名）
            if dplm_type is None:
                raise ValueError("`dplm_type` not found in config.json")

            net_class = get_net_class(dplm_type)
            net = net_class.from_pretrained(str(local_dir), **net_override)

            return cls(cfg=cfg_override, net=net)

    # 合并配置（与 DPLM 的 _update_cfg 完全一致）
    def _update_cfg(self, cfg):
        # # 原来的代码：
        # self.cfg = OmegaConf.merge(self._default_cfg, cfg)

        # 修改为：
        try:
            self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        except Exception as e:
            print(f"配置合并失败: {e}")
            print("使用文件配置，忽略默认配置")
            self.cfg = cfg  # 直接使用文件配置

    # 以下函数（q_sample_coupled / q_sample / forward / compute_loss / generate 等）
    # 基本保留 DPLM 的原始实现，逐行注释以便理解。
    # ---- q_sample_coupled ----
    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        # t1_eq_t2_mask 表示哪些序列的两个时间步相等（用于耦合策略）
        t1_eq_t2_mask = t1 == t2
        # 将 t1, t2 规整为 t1>=t2
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        # 对每个位置以概率（t1/num_timesteps）决定是否 mask（取代为 mask_id）
        t1_mask = (
            u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        # 将选中位置替换成 mask_id，得到 x_t1
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        # 在已经被 t1_mask 标记的位置，按比例决定是否保留在 t2
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        # 对于 t1==t2 的情况，按特殊规则处理
        t2_mask[t1_eq_t2_mask] = (
            u < (t1[t1_eq_t2_mask] / self.cfg.num_diffusion_timesteps)[:, None]
        ) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id)

        # 返回拼接后的结果：x_t (两个 batch 维度拼接)，t（时间步），以及 mask 掩码
        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0),
        }

    # ---- q_sample ----
    def q_sample(self, x_0, t1, maskable_mask):
        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (
            u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        # 注意：原 dplm 里有两次 masked_fill（可能是笔误或冗余），保留以兼容
        x_t1 = x_t1.masked_fill(t1_mask, self.mask_id)

        return {
            "x_t": x_t1,
            "t": t1,
            "mask_mask": t1_mask,
        }

    # ---- forward: 使用 net 产生 logits ----
    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        # net 的接口与 DPLM 保持一致：传入 input_ids，返回 dict，包含 "logits" 和可选的 "last_hidden_state"
        outputs = self.net(
            input_ids=input_ids,
        )
        logits = outputs["logits"]
        if return_last_hidden_state:
            last_hidden_state = outputs["last_hidden_state"]
            return logits, last_hidden_state
        else:
            return logits

    # ---- compute_loss: 训练时用的采样 + 损失计算逻辑 ----
    def compute_loss(self, batch, weighting="constant"):
        # batch 里期望含有 "targets"（即 ground truth token ids）
        """
        # print("=" * 80)
        # print("🟢 COMPUTE_LOSS FUNCTION START")
        # print(f"🔍 BATCH 类型: {type(batch)}")
        # #  <class 'dict'>
        # print(f"🔍 BATCH 内容:")
          Key: 'input_ids'
            Type: <class 'torch.Tensor'>
            Shape: torch.Size([2, 2048])
            Dtype: torch.int64
            Device: cuda:0
            Values[0, :5]: [142, 142, 124, 2, 91]
          Key: 'targets'
            Type: <class 'torch.Tensor'>
            Shape: torch.Size([2, 2048])
            Dtype: torch.int64
            Device: cuda:0
            Values[0, :5]: [142, 142, 124, 2, 91]
          Key: 'input_mask'
            Type: <class 'torch.Tensor'>
            Shape: torch.Size([2, 2048])
            Dtype: torch.bool
            Device: cuda:0
            Values[0, :5]: [True, True, True, True, True]
        """
        # print(type(batch), batch.keys() if isinstance(batch, dict) else batch.shape)
        target = batch["targets"]
        batch_size = target.size(0)
        """
        print("\n🟢 STEP 2: 采样时间步 t1, t2")
        print(f"  Batch size: {batch_size}")
        print(f"  num_diffusion_timesteps: {self.cfg.num_diffusion_timesteps}")
        Batch size: 2
        num_diffusion_timesteps: 500
        """
        # 随机采样两个时间步 t1, t2（长度为 2*B，随后 chunk 成两个向量）
        t1, t2 = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)
        """
        print(f"  t1 shape: {t1.shape}")
        print(f"  t2 shape: {t2.shape}")
        print(f"  t1 值: {t1.cpu().tolist()}")
        print(f"  t2 值: {t2.cpu().tolist()}")
          t1 shape: torch.Size([2])
          t2 shape: torch.Size([2])
          t1 值: [144, 410]
          t2 值: [299, 251]
        """
        # 如果启用 rdm_couple，则使用耦合样本策略（与论文/实现对应）
        if self.cfg.rdm_couple:
            print("  🔄 使用 q_sample_coupled 策略")
            x_t, t, loss_mask = list(
                self.q_sample_coupled(
                    target,
                    t1,
                    t2,
                    maskable_mask=self.get_non_special_symbol_mask(target),
                ).values()
            )
            print(f"    x_t shape: {x_t.shape}")
            print(f"    t shape: {t.shape}")
            print(f"    loss_mask shape: {loss_mask.shape}")
            # 目标也需要重复一次以匹配 x_t 的 batch 维度（因为耦合把 batch 翻倍）
            target = target.repeat(2, 1)
        else:
            # 否则使用普通 q_sample
            # print("  🔄 使用普通 q_sample 策略")
            x_t, t, loss_mask = list(
                self.q_sample(
                    target,
                    t1,
                    maskable_mask=self.get_non_special_symbol_mask(target),
                ).values()
            )
            """
            print(f"    x_t shape: {x_t.shape}")
            print(f"    t shape: {t.shape}")
            print(f"    loss_mask shape: {loss_mask.shape}")
            x_t shape: torch.Size([2, 2048])
            t shape: torch.Size([2])
            loss_mask shape: torch.Size([2, 2048])
            """
        """
        print("\n🟢 STEP 5: 扩散过程输出详细检查")
        print(f"  x_t (添加噪声后的token):")
        print(f"    shape: {x_t.shape}")
        print(f"    dtype: {x_t.dtype}")
        print(f"    示例值 (batch=0, first 10): {x_t[0, :10].cpu().tolist()}")
        print(f"    唯一值: {torch.unique(x_t).cpu().tolist()}")

        print(f"  t (时间步):")
        print(f"    shape: {t.shape}")
        print(f"    dtype: {t.dtype}")
        print(f"    值: {t.cpu().tolist()}")

        print(f"  loss_mask (损失mask):")
        print(f"    shape: {loss_mask.shape}")
        print(f"    dtype: {loss_mask.dtype}")
        print(f"    True的数量: {loss_mask.sum().item()}")
        print(f"    比例: {loss_mask.sum().item() / loss_mask.numel():.3f}")
        print(f"    示例 (batch=0, first 10): {loss_mask[0, :10].cpu().tolist()}")
        
        🟢 STEP 5: 扩散过程输出详细检查
          x_t (添加噪声后的token):
            shape: torch.Size([2, 2048])
            dtype: torch.int64
            示例值 (batch=0, first 10): [142, 374, 124, 2, 91, 11, 93, 142, 123, 142]
            唯一值: [0, 2, 11, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 102, 107, 116, 123, 124, 125, 126, 127, 128, 133, 141, 142, 265, 374]
          t (时间步):
            shape: torch.Size([2])
            dtype: torch.int64
            值: [144, 410]
          loss_mask (损失mask):
            shape: torch.Size([2, 2048])
            dtype: torch.bool
            True的数量: 2320
            比例: 0.566
            示例 (batch=0, first 10): [False, True, False, False, False, False, False, False, False, False]
        """
        # forward 得到 logits
        logits = self.forward(x_t)

        """
        

        print(f"  ✅ 正向传播完成")
        print(f"  logits shape: {logits.shape}")
        print(f"  logits dtype: {logits.dtype}")
        print(f"  logits device: {logits.device}")

        # 检查logits的有效性
        if torch.isnan(logits).any():
            print("  ⚠️ 警告: logits 包含 NaN 值!")
            print(f"    NaN 数量: {torch.isnan(logits).sum().item()}")

        if torch.isinf(logits).any():
            print("  ⚠️ 警告: logits 包含 Inf 值!")
            print(f"    Inf 数量: {torch.isinf(logits).sum().item()}")

        # 显示logits的统计信息
        print(f"  logits 统计:")
        print(f"    min: {logits.min().item():.6f}")
        print(f"    max: {logits.max().item():.6f}")
        print(f"    mean: {logits.mean().item():.6f}")
        print(f"    std: {logits.std().item():.6f}")
        
          ✅ 正向传播完成
              logits shape: torch.Size([2, 2048, 375])
              logits dtype: torch.float32
              logits device: cuda:0
              logits 统计:
                min: -3.312500
                max: 3.062500
                mean: -0.008304
                std: 0.738563
        """
        # 计算每个时间步的权重（linear 或 constant）
        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (
                num_timesteps - (t - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t),
        }[weighting][:, None].float() / num_timesteps
        """
        print(f"  权重计算完成:")
        print(f"    weight shape: {weight.shape}")
        print(f"    weight dtype: {weight.dtype}")
        print(f"    weight 值: {weight.squeeze().cpu().tolist()}")

        print(f"  返回的元组包含:")
        print(f"    1. logits: shape={logits.shape}")
        print(f"    2. target: shape={target.shape}")
        print(f"    3. loss_mask: shape={loss_mask.shape}")
        print(f"    4. weight: shape={weight.shape}")
          权重计算完成:
            weight shape: torch.Size([2, 1])
            weight dtype: torch.float32
            weight 值: [0.7140000462532043, 0.18200001120567322]
          返回的元组包含:
            1. logits: shape=torch.Size([2, 2048, 375])
            2. target: shape=torch.Size([2, 2048])
            3. loss_mask: shape=torch.Size([2, 2048])
            4. weight: shape=torch.Size([2, 1])
        """
        # 返回 logits, target, loss_mask 和权重（后续训练 loop 里会用这些来计算 loss）
        return logits, target, loss_mask, weight

    # ---- forward_encoder: 留空，按需扩展 ----
    def forward_encoder(self, input_tokens, **kwargs):
        # 如果需要 encoder-conditional generation，可以在子类覆盖此方法
        return {}

    # ---- initialize_output_tokens: 生成初始化的 output tokens（用 mask_id 填充需要预测的位置） ----
    def initialize_output_tokens(self, input_tokens, partial_masks=None, **kwargs):
        tokens = input_tokens
        if tokens is None:
            raise NotImplementedError
        else:
            # 得到可以被预测的位置掩码（非特殊符号）
            output_mask = self.get_non_special_symbol_mask(tokens, partial_masks=partial_masks)

            # 将这些位置替换为 mask_id，作为初始化
            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            # 初始化分数全为 0
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores

    # ---- resample: 用于 rejection sampling 去除重复 token 模式 ----
    def resample(self, _tokens, _scores, ratio, scale):
        """Rejection sampling to reduce repetitive tokens (e.g., 'VVVVV...')"""

        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []

        # 统计每个序列里各 token 的出现位置，找出频率最高的 token
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token_num = len(most_token_dict[token])
            # 如果某个 token 出现次数超过阈值（len(seq) * ratio），则把这些位置标为需要重采样
            if most_token_num > len(seq) * ratio:
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio:
                        mask |= seq.eq(k)
                resample_input_mask.append(mask)
                resample_input.append(seq.masked_fill(mask, self.mask_id))

        # 如果存在需要重采样的序列
        if len(to_be_resample_idx) > 0:
            # 把要重采样的序列堆成 batch 并转回相同 dtype
            resample_input = torch.stack(resample_input, dim=0).type_as(
                _tokens
            )
            resample_input_scores = torch.stack(
                resample_input_scores, dim=0
            ).type_as(_scores)
            resample_input_mask = (
                torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            )
            # 通过 net 重新预测 logits
            resample_logits = self.net(
                input_ids=resample_input,
            )["logits"]
            # 保证类型一致
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)
            # 把特殊 token 的 logits 设为 -inf，避免被采样
            resample_logits[..., self.mask_id] = -math.inf
#            resample_logits[..., self.x_id] = -math.inf
            resample_logits[..., self.pad_id] = -math.inf
            resample_logits[..., self.bos_id] = -math.inf
            resample_logits[..., self.eos_id] = -math.inf

            # 使用 top-k/top-p 过滤
            resample_logits = top_k_top_p_filtering(
                resample_logits, top_p=0.95
            )
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            (
                resample_tokens,
                resample_scores,
            ) = stochastic_sample_from_categorical(
                resample_logits, temperature=0.0, noise_scale=noise_scale
            )
            # 把重采样结果写回原始位置
            resample_input.masked_scatter_(
                resample_input_mask, resample_tokens[resample_input_mask]
            )
            resample_input_scores.masked_scatter_(
                resample_input_mask, resample_scores[resample_input_mask]
            )
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = (
                resample_input,
                resample_input_scores,
            )

    # ---- forward_decoder: decoder 步骤（用于生成） ----
    def forward_decoder(
        self,
        prev_decoder_out,
        encoder_out=None,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        disable_resample=True,
        resample_ratio=0.25,
    ):
        # 拷贝输入状态，避免原地修改影响调用方
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        # 计算当前可以预测的位置（非特殊符号）
        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        # 调用 net 得到 logits（模型的主接口）
        net_out = self.net(
            input_ids=output_tokens,
        )

        logits = net_out["logits"]
        attentions = net_out["attentions"] if need_attn_weights else None

        # 类型对齐：确保 logits 与 output_scores dtype 一致，方便后续比较/排序
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        # 将特殊 token 的 logits 设为 -inf，避免模型生成它们
        logits[..., self.mask_id] = -math.inf
#        logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf

        # 根据不同的采样策略选择 token
        if sampling_strategy == "vanilla":
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )
        elif sampling_strategy == "argmax":
            # 直接取最大概率
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            # 使用 Gumbel + argmax 的近似随机化采样
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )

            if not disable_resample:
                # 若允许重采样，则调用 rejection sampling 消除重复模式
                self.resample(
                    _tokens, _scores, ratio=resample_ratio, scale=1.0
                )
        else:
            raise NotImplementedError

        # 仅把预测位置填回去（masked_scatter_ 保证只替换 output_masks 位置）
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # 保存历史
        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,  # 可能包含注意力权重
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out.get("last_hidden_state", None),
        )

    # ---- get_non_special_symbol_mask: 计算非特殊 token 掩码 ----
    def get_non_special_symbol_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask

    # ---- _reparam_decoding: reparam 解码策略（复杂的 top-k / stochastic 策略实现） ----
    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        """This function is used to perform reparameterized decoding."""
        # decoding_strategy 格式: "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # 根据 schedule 计算去噪率 rate
        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # 计算用于 top-k 的 cutoff 长度 = 非特殊 token 数 * rate
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores)
            * rate
        ).long()
        # 将特殊 token 的分数设为较大值，避免被选中
        _scores_for_topk = cur_scores.masked_fill(
            ~non_special_sym_mask, 1000.0
        )

        # top-k 的两种模式：stochastic (带 Gumbel 噪声) 或 deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(
                _scores_for_topk,
                cutoff_len,
                stochastic=True,
                temp=noise_scale * rate,
            )
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(
                _scores_for_topk, cutoff_len, stochastic=False
            )
        else:
            raise NotImplementedError

        # 依据 condition（cond/uncond）计算 not_v1_t，与 top-k 策略相关
        if condition == "cond":
            not_v1_t = (
                (cur_tokens == output_tokens)
                & (cur_scores < output_scores)
                & lowest_k_mask
            )
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # 对 b_t = 0 的位置做处理（若在 lowest_k 中则置为噪声）
        not_v2_t = lowest_k_mask

        last_mask_position = xt_neq_x0
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        # 将被 mask_to_noise 的位置赋值为 noise（tensor 或 scalar）
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(
                masked_to_noise, noise[masked_to_noise]
            )
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError(
                "noise should be either a tensor or a scalar"
            )
        # 把对应位置的分数设置为 -inf
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        # masked_to_x0 表示置为当前 cur_tokens 的位置
        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()

        # 计算并返回下一个 not_b_t（为下一步保存）
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores

    # ---- generate: 高层生成循环（调用初始化、decoder step、reparam 策略） ----
    def generate(
        self,
        input_tokens,
        tokenizer=None,
        max_iter=None,
        temperature=None,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        disable_resample=False,
        resample_ratio=0.25,
    ):
        # 保持接口风格：传入 tokenizer / max_iter / temperature 等，默认行为与 DPLM 一致
        tokenizer = tokenizer
        max_iter = max_iter
        temperature = temperature

        # 0) encoder（可选）
        encoder_out = self.forward_encoder(input_tokens)
        # 1) 初始化 output tokens（用 mask 填充需要预测的位置）
        (
            initial_output_tokens,
            initial_output_scores,
        ) = self.initialize_output_tokens(
            input_tokens, encoder_out=encoder_out, partial_masks=partial_masks
        )
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        # 计算初始的 output_masks
        prev_decoder_out["output_masks"] = self.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        # 迭代 decoding 步骤
        # for step in tqdm(range(max_iter), desc="Decoding"):
        for step in range(max_iter):

            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    disable_resample=disable_resample,
                    resample_ratio=resample_ratio,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]

            # 2.2: 对低置信度部分重新掩码并使用 reparam 解码策略
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )

            (
                output_masks,
                result_tokens,
                result_scores,
            ) = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy="reparam-uncond-deterministic-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_id,
            )

            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out["history"],
            )

        decoder_out = prev_decoder_out
        # 返回最终生成 token 矩阵
        return decoder_out["output_tokens"]
