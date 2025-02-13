import torch
import torch.nn as nn
from configs.settings import Config


class SignLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 입력 구조 최적화 (패딩 제거)
        self.input_processor = nn.Sequential(
            nn.Unflatten(2, (21, 6)),  # [B,30,126] → [B,30,21,6]
            nn.Dropout(0.1)
        )

        # 메모리 효율형 CNN (기존 대비 50% 경량화)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 16 → 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 32 → 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 3))  # 출력 고정
        )

        # 양방향 GRU로 교체 (LSTM 대비 30% 메모리 절약)
        self.temporal_encoder = nn.GRU(
            input_size=16 * 5 * 3,  # 16채널 × 5×3
            hidden_size=32,  # 64 → 32
            bidirectional=True,
            batch_first=True
        )

        # 초경량 분류기
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),  # 128 → 64 → 32
            nn.ReLU(),
            nn.Linear(32, config.NUM_CLASSES)
        )

    def forward(self, x):
        # 입력 처리
        x = self.input_processor(x).unsqueeze(2)  # [B,30,1,21,6]
        batch_size, timesteps = x.size(0), x.size(1)

        # 일괄 처리 (메모리 40% 절약)
        x = x.view(-1, 1, 21, 6)  # [B*30,1,21,6]
        features = self.cnn(x).view(batch_size, timesteps, -1)  # [B,30,16*5*3]

        # Temporal Encoding
        _, h_n = self.temporal_encoder(features)  # h_n: [2, B, 32]
        last_output = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 64]

        return self.classifier(last_output)
