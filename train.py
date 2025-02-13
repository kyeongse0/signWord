import torch
import time
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix
from configs.settings import Config
from data.data_loader import TorchDataLoader
from models.model_builder import SignLanguageModel
from utils.callbacks import get_training_utils
import os


def main():
    # 1. 메모리 설정
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    torch.mps.set_per_process_memory_fraction(0.7)

    # 2. 환경 초기화
    config = Config()
    device = torch.device("mps")
    start_time = time.time()

    # 3. 데이터 로드
    loader = TorchDataLoader(config)
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )

    # 4. 모델 초기화
    model = SignLanguageModel(config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-5
    )
    checkpoint, early_stop = get_training_utils(config)
    scaler = GradScaler(enabled=True)

    # 5. 학습 루프
    best_val_acc = 0.0
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        # 배치별 진행률 표시
        batch_bar = tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{config.EPOCHS}]",
                         bar_format="{l_bar}{bar:20}{r_bar}")
        for batch_idx, (inputs, labels) in enumerate(batch_bar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).argmax(dim=1)

            # 혼합 정밀도 학습
            with autocast(device_type='mps'):
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)

            # 역전파 및 최적화
            scaler.scale(loss).backward()
            if (batch_idx + 1) % 2 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            batch_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # 6. 검증 단계
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.argmax(dim=1).to(device)

                outputs = model(inputs)
                val_loss += nn.functional.cross_entropy(outputs, labels).item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        # 7. 성능 지표 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        epoch_time = time.time() - epoch_start

        # 8. 체크포인트 및 진행 상황 출력
        checkpoint(val_acc, model)
        print(f"\n⏳ Epoch {epoch + 1}/{config.EPOCHS} | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        print(f"Best Acc: {checkpoint.best_score:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if early_stop(val_loss):
            print(f"🛑 Early Stopping at epoch {epoch + 1}")
            break

    # 9. 최종 평가
    model.load_state_dict(torch.load(config.SAVE_DIR / "normalized_480_best_model.pth"))
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.argmax(dim=1).to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 10. 평가 결과 출력
    print("\n🎯 Final Test Results")
    print(f"Accuracy: {(np.array(all_preds) == np.array(all_labels)).mean():.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=config.CLASSES))
    print(f"\nTotal Training Time: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
