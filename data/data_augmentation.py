import numpy as np
import random
import os

train_data_dir = "../train_data"


def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def time_warp(data, warp_factor=0.1):
    factor = 1 + random.uniform(-warp_factor, warp_factor)
    indices = np.linspace(0, data.shape[0] - 1, int(data.shape[0] * factor)).astype(int)
    indices = np.clip(indices, 0, data.shape[0] - 1)
    return data[indices]


def interpolate_data(data, target_length=30):
    original_length = data.shape[0]
    new_data = np.zeros((target_length, data.shape[1]))
    for i in range(data.shape[1]):
        new_data[:, i] = np.interp(
            np.linspace(0, original_length - 1, target_length),
            np.arange(original_length),
            data[:, i]
        )
    return new_data


def augment_data(data, num_augments=5):
    augmented_data = []
    for _ in range(num_augments):
        aug = data.copy()
        if random.random() < 0.5:
            aug = add_noise(aug)
        if random.random() < 0.5:
            aug = time_warp(aug)
        aug = interpolate_data(aug)  # Ensure final shape consistency
        augmented_data.append(aug)
    return np.array(augmented_data)


# 데이터 불러와서 증강 후 저장
for class_name in os.listdir(train_data_dir):
    class_path = os.path.join(train_data_dir, class_name)

    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        data = np.load(file_path)
        augmented_samples = augment_data(data, num_augments=5)

        for i, aug_data in enumerate(augmented_samples):
            aug_file_name = f"{os.path.splitext(file_name)[0]}_augmented_{i}.npy"
            aug_file_path = os.path.join(class_path, aug_file_name)
            np.save(aug_file_path, aug_data)
