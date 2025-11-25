import torch
import numpy as np
from torch.utils.data import Sampler
from verl.experimental.dataset.sampler import AbstractSampler
from collections import defaultdict
import random

class ClassGroupedSequentialSampler(AbstractSampler):
    """
    使得每个 batch 内数据来自同一类别，
    不同 batch 属于不同类别（相邻 batch 的类别不同）。
    数据类别由 dataset[i]['data_source'] 字段决定。
    """

    def __init__(self, data_source, data_config, shuffle_classes=True, shuffle_within_class=True, seed=None):
        """
        Args:
            data_source: 数据集对象 (支持 dataset[i]['data_source'] 访问)
            batch_size: 每个 batch 的大小
            shuffle_classes: 是否随机打乱类别顺序
            shuffle_within_class: 是否在每个类别内随机打乱样本顺序
            seed: 随机种子，确保可复现
        """
        self.data_source = data_source
        self.batch_size = data_config.train_batch_size
        self.shuffle_classes = shuffle_classes
        self.shuffle_within_class = shuffle_within_class
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        # 根据 data_source 分组
        self.class_to_indices = defaultdict(list)
        for idx, item in enumerate(self.data_source):
            cls = item["data_source"]
            self.class_to_indices[cls].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.sorted_indices = self._prepare_epoch_indices()

    def _prepare_epoch_indices(self):
        """生成一个 epoch 的样本顺序"""
        # Step 1: 先构建每个类别的 batch 列表
        class_batches = {}
        for cls in self.classes:
            indices = self.class_to_indices[cls]
            if self.shuffle_within_class:
                indices = torch.tensor(indices)
                indices = indices[torch.randperm(len(indices), generator=self.generator)].tolist()
            # 分成batch
            class_batches[cls] = [
                indices[i:i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]

        # Step 2: 将所有类别的batch混合（保证相邻batch不同类）
        batches = []
        class_batch_lists = {cls: lst.copy() for cls, lst in class_batches.items()}
        available_classes = self.classes.copy()
        if self.shuffle_classes:
            random.Random(self.generator.initial_seed()).shuffle(available_classes)

        last_class = None
        while any(len(v) > 0 for v in class_batch_lists.values()):
            # 从可选类别中选择（排除上一个类）
            valid_classes = [c for c in available_classes if c != last_class and len(class_batch_lists[c]) > 0]
            if not valid_classes:  # 若剩下的类别都用完了，允许重复上一个类别
                valid_classes = [c for c in available_classes if len(class_batch_lists[c]) > 0]
            chosen_class = random.choice(valid_classes)

            batch = class_batch_lists[chosen_class].pop(0)
            batches.append((chosen_class, batch))
            last_class = chosen_class

        # Step 3: 拼接成完整索引序列
        epoch_indices = []
        for _, batch in batches:
            epoch_indices.extend(batch)
        return epoch_indices

    def update(self):
        """重新生成新的顺序"""
        self.sorted_indices = self._prepare_epoch_indices()

    def __iter__(self):
        """返回一个 epoch 的完整顺序"""
        return iter(self.sorted_indices)

    def __len__(self):
        """返回样本总数"""
        return len(self.sorted_indices)
