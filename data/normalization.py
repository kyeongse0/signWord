import numpy as np
import os
from tqdm import tqdm


def normalize_landmarks(hand_data):
    """1차원 손 데이터 정규화 (63차원 입력)"""
    if hand_data.size != 63 or np.all(hand_data == 0):
        return np.zeros_like(hand_data)

    # 1차원 → (21,3) 변환
    landmarks = hand_data.reshape(21, 3)

    wrist = landmarks[0]
    palm_vector = landmarks[9] - wrist
    scale_factor = np.linalg.norm(palm_vector)

    if scale_factor < 1e-7:
        return hand_data

    normalized = (landmarks - wrist) / scale_factor
    return normalized.flatten()  # 63차원으로 복원


def process_dataset(input_dir, output_dir):
    """차원 문제 해결 버전"""
    os.makedirs(output_dir, exist_ok=True)

    for cls in tqdm(os.listdir(input_dir)):
        input_cls_path = os.path.join(input_dir, cls)
        output_cls_path = os.path.join(output_dir, cls)
        os.makedirs(output_cls_path, exist_ok=True)

        for file in os.listdir(input_cls_path):
            if file.endswith('.npy'):
                data = np.load(os.path.join(input_cls_path, file))  # (30, 126)

                processed = []
                for frame in data:
                    # 손 분할 (63차원씩)
                    hands = [frame[:63], frame[63:]]

                    norm_hands = []
                    for hand in hands:
                        norm_hands.append(normalize_landmarks(hand))

                    processed.append(np.concatenate(norm_hands))

                np.save(os.path.join(output_cls_path, file), np.array(processed))


# 실행
process_dataset('../train_data', 'normalized_data')
