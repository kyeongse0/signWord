import numpy as np
import os

DATA_PATH = 'output_npy'
GESTURE = '실패'  # 확인할 제스처
gesture_path = os.path.join(DATA_PATH, GESTURE)

# 모든 .npy 파일 검사
for file in os.listdir(gesture_path):
    if file.endswith('.npy'):
        file_path = os.path.join(gesture_path, file)
        data = np.load(file_path)

        if data.shape != (30, 126):
            print(f"⚠️ {file}의 shape가 이상함: {data.shape}")
        else:
            print(f"✅ {file} - 정상 (30, 63)")
            print("첫번째 프레임 데이터\n",  data[0])
