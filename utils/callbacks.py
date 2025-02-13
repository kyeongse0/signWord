import torch
import os
from pathlib import Path


class Checkpoint:
    def __init__(self, config, monitor='val_acc', mode='max'):
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.mode = mode
        self.monitor = monitor
        self.save_path = Path(config.SAVE_DIR) / "normalized_480_best_model.pth"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, current, model):
        if (self.mode == 'max' and current > self.best_score) or \
                (self.mode == 'min' and current < self.best_score):
            self.best_score = current
            torch.save(model.state_dict(), self.save_path)
            print(f"\n✅ Best model saved: {self.monitor}={current:.4f}")


class EarlyStopper:
    def __init__(self, patience=5, delta=0, mode='min'):
        self.patience = patience
        self.delta = abs(delta)
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, current):
        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"⛔ Early stopping after {self.patience} epochs without improvement")
                return True
        return False

    def _is_improvement(self, current):
        if self.mode == 'min':
            return current < (self.best_value - self.delta)
        return current > (self.best_value + self.delta)

def get_training_utils(config):
    checkpoint = Checkpoint(config, monitor='val_acc', mode='max')
    early_stop = EarlyStopper(
        patience=config.PATIENCE,
        delta=0.001,
        mode='min'
    )
    return checkpoint, early_stop
