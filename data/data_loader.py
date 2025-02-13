import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
import random  # 추가된 임포트


class TorchDataLoader:
    def __init__(self, config):
        self.config = config
        self.classes = config.CLASSES
        self.data_root = Path(config.DATA_ROOT)
        self.batch_size = config.BATCH_SIZE

    def _load_class_data(self, class_dir: Path, max_samples: int = 480) -> torch.Tensor:
        all_files = list(class_dir.glob('*.npy'))

        # 경로 존재 여부 검증 추가
        if not class_dir.exists():
            raise FileNotFoundError(f"Directory {class_dir} not found")

        # 파일 개수 검증 (심화)
        num_files = len(all_files)
        if num_files == 0:
            # 실제 누락된 파일 목록 출력
            npy_in_parent = list(class_dir.parent.glob('*.npy'))
            raise ValueError(
                f"No .npy files in {class_dir}. Found {len(npy_in_parent)} files in parent."
                "Check directory structure or file extensions."
            )

        # 랜덤 셔플 (재현성을 위한 시드 설정)
        random.seed(42)
        random.shuffle(all_files)

        # 최대 240개 샘플 선택
        selected_files = all_files[:min(max_samples, num_files)]

        samples = []
        for file in selected_files:  # 수정된 부분
            arr = np.load(file)
            samples.append(torch.from_numpy(arr))

        return torch.stack(samples)


    def load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """전체 데이터셋 로드 및 텐서 변환"""
        features, labels = [], []

        for cls_idx, cls_name in enumerate(self.classes):
            class_dir = self.data_root / cls_name
            class_samples = self._load_class_data(class_dir)
            features.append(class_samples)
            labels.extend([cls_idx] * len(class_samples))

        X = torch.cat(features, dim=0).float()
        y = torch.nn.functional.one_hot(
            torch.tensor(labels),
            num_classes=self.config.NUM_CLASSES
        ).float()

        return X, y

    def get_splits(self):
        """데이터 분할 및 DataLoader 생성"""
        X, y = self.load_dataset()

        # 1차 분할: Train(70%) / Temp(30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X.numpy(), y.numpy(),  # sklearn을 위해 numpy 변환
            test_size=1 - self.config.SPLIT_RATIO[0],
            stratify=y.numpy(),
            random_state=42
        )

        # 다시 텐서로 변환
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        X_temp, y_temp = torch.from_numpy(X_temp), torch.from_numpy(y_temp)

        # 2차 분할: Val(20%) / Test(10%)
        val_test_ratio = self.config.SPLIT_RATIO[2] / (self.config.SPLIT_RATIO[1] + self.config.SPLIT_RATIO[2])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp.numpy(), y_temp.numpy(),
            test_size=val_test_ratio,
            stratify=y_temp.numpy(),
            random_state=42
        )

        # 최종 텐서 변환
        X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
        X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_dataloaders(self, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.get_splits()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

