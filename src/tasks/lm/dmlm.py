# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
该文件定义了 DMLM（Crystal Domain Masked Language Model）的任务训练逻辑，
整体结构与 dplm.py 基本一致，只是任务注册名和模型替换为了 DMLM。
"""

from typing import Any, Union

import torch
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import MeanMetric, MinMetric

from byprot import utils
from byprot.tasks import TaskLitModule, register_task
from byprot.utils.config import compose_config as Cfg
import os
# 获取日志句柄
log = utils.get_logger(__name__)


def new_arange(x, *size):
    """生成一个和输入张量 x 在同一设备上的范围序列张量。
    参数:
        x: 参考张量，用于确定设备
        size: 指定生成的形状，如果为空，则使用 x 的 size()
    返回:
        [*size] 大小的 Tensor，最后一维是 arange 序列
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


# ----------------------------- #
# 任务类定义：DMLMTrainingTask
# ----------------------------- #
@register_task("lm/dmlm")  # 注册任务名为 lm/dmlm
class DMLMTrainingTask(TaskLitModule):
    """DMLM 的 Lightning 模块封装，用于训练和评估"""

    # 默认配置
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise="rdm",  # ['full_mask', 'random_mask']
            num_unroll=0,
            watch_t1_t2_loss=False,
            cal_constant_loss=False,
            weight="constant",
        ),
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],        # 模型配置或实例
        criterion: Union[nn.Module, DictConfig],    # 损失函数配置或实例
        optimizer: DictConfig,                      # 优化器配置
        lr_scheduler: DictConfig = None,            # 学习率调度器配置
        *,
        learning=_DEFAULT_CFG.learning,             # 训练相关配置
    ):
        # if not hasattr(self.hparams, "model"):
        #     log.warning("[Warning] self.hparams.model not found! Using default model config.")
        #     self.hparams.model = {}
        log.info(f'Function lm/dmlm.py.__init__() Start!')
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # 保存超参数，方便日志和断点恢复
        self.save_hyperparameters(logger=True)

        # 构建模型
        log.info(f'Function lm/dmlm.py.__init__() step 1: build model!')
        self.build_model()
        # 提取分词器（来自模型内部）
        log.info(f'Function lm/dmlm.py.__init__() step 2: build tokenizer!')
        self.tokenizer = self.model.tokenizer
        self.loss_accumulator = []
        log.info(f'Function lm/dmlm.py.__init__() Done!')


    def setup(self, stage=None) -> None:
        """在不同阶段的初始化逻辑"""
        super().setup(stage)

        # 构建损失函数
        self.build_criterion()
        # 构建评估指标
        self.build_torchmetric()

        if self.stage == "fit":
            log.info(f"\n{self.model}")  # 打印模型结构
        elif self.stage == "test":
            self.test_step_outputs = []  # 存储测试输出

    def on_before_optimizer_step(self, optimizer):
        """在优化器更新前记录梯度范数"""
        if self.global_rank == 0:  # 仅在主进程上执行
            grad_norm_dict = grad_norm(
                self.trainer.strategy.model, norm_type=2
            )
            self.log_dict(grad_norm_dict)

    def build_model(self):
        """根据配置构建 DMLM 模型"""
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(
            cfg=self.hparams.model, group="model"
        )

    def build_criterion(self):
        """根据配置构建损失函数，并设置 ignore_index"""
        self.criterion = utils.instantiate_from_config(
            cfg=self.hparams.criterion
        )
        # pad_token 不计入损失
        self.criterion.ignore_index = self.tokenizer.pad_token_id

    def build_torchmetric(self):
        """构建评估指标（loss 和 ppl）"""
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()
        self.val_ppl_best = MinMetric()

    def step(self, batch):
        """一次前向和损失计算逻辑
        batch 是一个字典，包含：
            - coords: [bsz, len, n_atoms, 3], 原子坐标
            - coord_mask: [bsz, len], 有效坐标 mask
            - lengths: [bsz, len], 序列长度
            - tokens: [bsz, len], token 序列
        """
        weighting = self.hparams.learning.weight
        # 模型计算 logits, target, loss_mask, weights
        logits, target, loss_mask, weights = self.model.compute_loss(
            batch, weighting=weighting
        )
        # todo: 输出logits和target，使用decoder输出一下logits和target看看有没有什么问题
        # print('Function Step: start decoding and saving to 0!')
        # self.test_decode_save(logits)
        # self.test_decode_save(target)
        # 使用 criterion 计算损失
        loss, logging_output = self.criterion(
            logits,
            target,
            loss_mask,
            weights,
            watch_t1_t2_loss=self.hparams.learning.watch_t1_t2_loss,
            cal_constant_loss=self.hparams.learning.cal_constant_loss,
        )

        # 检查 NaN，避免训练崩溃
        if torch.isnan(loss):
            print("Loss NAN on step ", self.global_step)
            loss = loss * 0
            logging_output["nll_loss"] = logging_output["nll_loss"] * 0
            logging_output["fullseq_loss"] = logging_output["fullseq_loss"] * 0
            logging_output["fullseq_nll_loss"] = (
                logging_output["fullseq_nll_loss"] * 0
            )
            logging_output["ppl"] = logging_output["ppl"] * 0
        # print(f"Total Loss: {loss.item():.6f}")
        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):

        loss, logging_output = self.step(batch)
        # print(f"Step {self.global_step}正常！")
        # # 打印详细的 loss 信息
        # if batch_idx % 100 == 0:
        #     print(f"\n=== Batch {batch_idx} (Step {self.global_step}) ===")
        #     print(f"Total Loss: {loss.item():.6f}")
        self.loss_accumulator.append(loss.item())
        # 更新1000次生成一次
        if self.global_step % 1000 == 0:
            avg_loss = sum(self.loss_accumulator) / len(self.loss_accumulator)
            print(f"\n=== 每1000次更新的平均Loss ===")
            print(f"Step {self.global_step}, 最近1000个step的平均Loss: {avg_loss:.6f}")
            print(f"样本数量: {len(self.loss_accumulator)}")
            print("=" * 40)
            self.test_generate(batch_idx)
            # 重置累积器
            self.loss_accumulator = []

        # 记录训练过程指标
        self.log("global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lr", self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(
                f"train/{log_key}",
                log_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return {"loss": loss}

    # -------------------- #
    # 验证和测试逻辑
    # -------------------- #

    def test_decode_save(self, samples, batch_idx=-1):
        decoded_seqs = self.model.tokenizer.batch_decode(samples.tolist())
        clean_seqs = [''.join(seq.split(' ')) for seq in decoded_seqs]

        # 保存到文件
        filename = f"cif_results/generated_batch_-2.txt"
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"=== Batch {batch_idx} 生成的序列 ===\n\n")
            for i, seq in enumerate(clean_seqs):
                f.write(f"序列 {i + 1} (长度: {len(seq)}):\n")
                f.write(f"{seq}\n\n")

        print(f"✅ 结果已保存到: {filename}")
        print(f"生成样本数量: {len(clean_seqs)}")

    def test_generate(self, batch_idx: int):
        device = self.device  # Lightning模块有device属性

        input_tokens = torch.full((5, 400), 374, device=device, dtype=torch.long)

        # 使用 dmlm 生成
        samples = self.model.generate(
            input_tokens=input_tokens,
            max_iter=500,
        )
        self.test_decode_save(samples, batch_idx)
        # 解码并打印结果
        # 解码结果

        # 仍然在控制台打印前两个样本
        # for i, seq in enumerate(clean_seqs[:2]):
        #     print(f"示例序列 {i + 1}: {seq[:100]}...")

    def on_test_epoch_start(self) -> None:
        """在测试开始时，设置 noise 策略为 full_mask"""
        self.hparams.noise = "full_mask"

    def validation_step(self, batch: Any, batch_idx: int):
        """验证时的单步逻辑"""
        import time
        from datetime import datetime

        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(f'[{current_time}] Start validation_step!')
        loss, logging_output = self.step(batch)

        # 累积评估指标
        sample_size = logging_output["sample_size"]
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(logging_output["nll_loss"], weight=sample_size)
        # print('Validation_step End!')
        return {"loss": loss}

    def on_validation_epoch_end(self):
        """验证或测试结束时，计算并记录最终指标"""
        log_key = "test" if self.stage == "test" else "val"

        # 计算整个验证集的平均 loss 和 ppl
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        # 记录指标
        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        # 如果是训练阶段，还要更新最优 ppl
        if self.stage == "fit":
            self.val_ppl_best.update(eval_ppl)
            self.log(
                "val/ppl_best",
                self.val_ppl_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )

        super().on_validation_epoch_end()
