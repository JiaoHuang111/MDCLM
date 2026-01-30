# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from byprot.tokenizers.crysta_tokenizer import CrystaTokenizerWrapper
# ✅ 导入注册装饰器
from byprot.datamodules import register_datamodule
from byprot import utils

log = utils.get_logger(__name__)

class CrystalDataset(Dataset):
    """Dataset for tokenized CIFs (train.bin / val.bin / meta.pkl)."""

    def __init__(self, data_file: str = "/data/huangjiao/dmlm/data-bin/crystalmols",
                 max_len: int = 2048,
                 pad_token_id: int = 376):
        log.info(f'Function CrystalDataset.__init__() Start.')
        super().__init__()
        self.max_len = max_len
        # self.tokenizer = tokenizer
        # self.pad_token_id = pad_token_id

        # load tokenized sequences (uint16 array)
        if data_file == "/data/huangjiao/dmlm/data-bin/crystalmols/train.bin":
            start_file = "/data/huangjiao/dmlm/data-bin/crystalmols/starts_v1_train.pkl"
        elif data_file == "/data/huangjiao/dmlm/data-bin/crystalmols/val.bin":
            start_file = "/data/huangjiao/dmlm/data-bin/crystalmols/starts_v1_val.pkl"
        else:
            start_file = None
        with open(start_file, "rb") as f:
            start_list = pickle.load(f)
            self.start_data = np.array(start_list, dtype=np.int64)

        with open(data_file, "rb") as f:
            self.samples = np.frombuffer(f.read(), dtype=np.uint16)
        self.num_samples = len(self.start_data) - 1
        # split into sequences
        # assumes CrystaLLM stores [len, tokens...] format
        # self.seq_starts = []
        # self.pad_token_id = pad_token_id
        # i = 0
        # while i < len(self.samples):
        #     length = self.samples[i]
        #     self.seq_starts.append(i)
        #     i += length + 1
        log.info(f'Function CrystalDataset.__init__() Done.')

    def __len__(self):
        return self.num_samples - 1

    def __getitem__(self, idx):

        start = self.start_data[idx]
        end = self.start_data[idx + 1]

        data_slice = self.samples[start:end]
        # 关键：使用np.array()创建完全独立的副本
        tokens_np = np.array(data_slice, dtype=np.int64)  # np.array()创建新数组

        # 使用torch.tensor()而不是torch.from_numpy()
        # torch.tensor()会创建独立副本，torch.from_numpy()可能共享内存
        tokens = torch.tensor(tokens_np, dtype=torch.long)

        # 确保张量是连续的
        tokens = tokens.contiguous()
        return {
            "input_ids": tokens,
            "targets": tokens.clone(),
            "input_mask": torch.ones(len(tokens), dtype=torch.bool).contiguous()
        }
        # return torch.tensor(tokens, dtype=torch.long)


@register_datamodule("crystalmols_hf")
class CrystalMolsDataModule(pl.LightningDataModule):
    """Lightning DataModule for CrystaLLM tokenized CIF datasets."""

    def __init__(
            self,
            data_dir: str = "/data/huangjiao/dmlm/crystalmols",
            max_tokens=8000,
            max_len=2048,
            num_workers=8,
            tokenizer="crystallm",
            vocab_file=None,
            batch_size=8,
            special_tokens=None,  # ✅ 接收多余的参数
            **kwargs,  # ✅ 安全接收所有其他意外参数（防止Hydra配置出错）
    ):
        log.info(f'Function CrystalMolsDataModule__init__ Start!')
        super().__init__()

        # ✅ 确保 data_dir 是字符串
        log.info(f'Function CrystalMolsDataModule__init__: Step 1: Initialize parameter!')
        if not isinstance(data_dir, (str, os.PathLike)):
            from omegaconf import OmegaConf
            try:
                data_dir = OmegaConf.to_container(data_dir, resolve=True)
                if isinstance(data_dir, dict):
                    # 如果 data_dir 是 {"path": "..."} 结构，取其中路径
                    data_dir = list(data_dir.values())[0]
                data_dir = str(data_dir)
            except Exception:
                data_dir = str(data_dir)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.max_len = max_len
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.special_tokens = special_tokens  # ✅ 保存但不使用（防止冲突）

        log.info(f'Function CrystalMolsDataModule__init__: Step 2: Load meta.pkl!')
        # Load meta.pkl
        meta_file = os.path.join(self.data_dir, "meta.pkl")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"meta.pkl not found in {self.data_dir}")
        with open(meta_file, "rb") as f:
            self.meta = pickle.load(f)

        log.info(f'Function CrystalMolsDataModule__init__: Step 3: Initialize tokenizer!')
        # Initialize tokenizer
        if tokenizer == "crystallm":
            self.tokenizer = CrystaTokenizerWrapper(meta=self.meta)
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer}")

        # Special token ids from meta.pkl
        self.pad_token_id = self.meta.get("pad_token_id", 371)
        self.eos_token_id = self.meta.get("eos_token_id", 372)
        self.vocab_size = self.meta.get("vocab_size", 375)
        self.train_dataset = None  # 缓存Dataset
        self.val_dataset = None
        log.info(f'Function CrystalMolsDataModule__init__ Done!')

    def setup(self, stage=None):
        if self.train_dataset is None:
            train_file = os.path.join(self.data_dir, "train.bin")
            self.train_dataset = CrystalDataset(train_file, self.max_len, self.pad_token_id)
        if self.val_dataset is None:
            val_file = os.path.join(self.data_dir, "val.bin")
            self.val_dataset = CrystalDataset(val_file, self.max_len, self.pad_token_id)

    def _create_collate_fn(self):
        """创建安全的collate函数"""

        def collate_fn(batch):
            if not batch:
                return {}

            # 获取最大序列长度
            max_len = max(item["input_ids"].size(0) for item in batch)
            batch_size = len(batch)

            # 创建全新的张量
            input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
            targets = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

            for i, item in enumerate(batch):
                seq_len = item["input_ids"].size(0)
                input_ids[i, :seq_len] = item["input_ids"]
                targets[i, :seq_len] = item["targets"]
                attention_mask[i, :seq_len] = True

            return {
                "input_ids": input_ids,
                "targets": targets,
                "attention_mask": attention_mask
            }

        return collate_fn

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        collate_fn = self._create_collate_fn()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,  # 预取数据
            collate_fn=collate_fn,  # 同样使用
            drop_last=True,  # 丢弃最后不完整的批次
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup()
        collate_fn = self._create_collate_fn()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,  # 预取数据
            collate_fn=collate_fn,  # 同样使用
        )

