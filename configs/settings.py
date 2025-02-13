import os
from pathlib import Path


class Config:
    # 데이터 설정
    DATA_ROOT = Path("normalized_data")
    DATA_ROOT.mkdir(exist_ok=True)  # 디렉토리 강제 생성

    # 클래스 필터링 로직 강화
    CLASSES = sorted([d.name for d in DATA_ROOT.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])

    INPUT_SHAPE = (30, 126)
    NUM_CLASSES = len(CLASSES)
    BATCH_SIZE = 16
    SPLIT_RATIO = [0.7, 0.2, 0.1]

    # 모델 설정
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    LSTM_UNITS = [128, 64]

    # 트레이닝 설정
    EPOCHS = 30
    PATIENCE = 15

    # 저장 경로 설정
    SAVE_DIR = Path("models/saved_models")
    SAVE_DIR.mkdir(exist_ok=True)  # 모델 저장 디렉토리 생성
