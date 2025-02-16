import cv2
import mediapipe as mp
import numpy as np
import time
import torch
import torch.nn as nn
from configs.settings import Config
from PIL import Image, ImageDraw, ImageFont

# 사용할 한글 폰트 경로 (AppleSDGothicNeo)
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"

def put_text(img, text, pos, font_path, font_size=32, color=(0, 0, 255)):

    # OpenCV BGR 이미지를 PIL RGB 이미지로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))  # PIL은 RGB 순서 사용
    # 다시 OpenCV 이미지(BGR)로 변환
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ----- 모델 및 클래스 정의  -----
class SignLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super(SignLanguageModel, self).__init__()
        self.config = config

        # 입력 처리: [B, 30, 126] -> [B, 30, 21, 6]
        self.input_processor = nn.Sequential(
            nn.Unflatten(2, (21, 6)),
            nn.Dropout(0.1)
        )

        # 메모리 효율형 CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 3))
        )

        # 양방향 GRU
        self.temporal_encoder = nn.GRU(
            input_size=16 * 5 * 3,
            hidden_size=32,
            bidirectional=True,
            batch_first=True
        )

        # 초경량 분류기
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.NUM_CLASSES)
        )

    def forward(self, x):
        # x: [B, 30, 126] → [B, 30, 21, 6] 후 추가 차원 삽입: [B, 30, 1, 21, 6]
        x = self.input_processor(x).unsqueeze(2)
        batch_size, timesteps = x.size(0), x.size(1)
        # CNN 연산을 위해 배치와 타임스텝 결합: [B*30, 1, 21, 6]
        x = x.view(-1, 1, 21, 6)
        features = self.cnn(x).view(batch_size, timesteps, -1)
        _, h_n = self.temporal_encoder(features)
        last_output = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 64]
        return self.classifier(last_output)

# config 객체를 생성하여 모델 생성 시 인자로 전달
config = Config()

# 학습 시 사용한 클래스(제스처) 설정
GESTURES = {idx: class_name for idx, class_name in enumerate(config.CLASSES)}
#GESTURES = { 29:'실패'}

model = SignLanguageModel(config)
model.load_state_dict(torch.load("models/saved_models/train_480_best_model.pth", map_location=torch.device('cpu')))
model.eval()  # 평가 모드

# ----- Mediapipe 셋업 -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 두 손 모두 검출하도록 설정
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# 🔥 버퍼 사이즈 설정이 추가된 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 1로 설정

# 상태 관리 변수들
phase = "countdown"
phase_start_time = time.time()
captured_frames = []
current_problem_idx = 0
attempts_left = 3
total_results = []
PROBLEMS = list(GESTURES.values())

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 획득 실패! 재시도...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        continue

    # 미디어파이프 처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 화면 표시 로직 (기존과 동일)
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks,
                                                               results.multi_handedness)):
            label = handedness.classification[0].label
            frame = put_text(frame, f"Hand {idx + 1}: {label}", (10, 50 + idx * 30),
                             FONT_PATH, font_size=24, color=(0, 255, 0))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()

    # 🔥 핵심 수정 부분: 데이터 수집 로직 -------------------------------------------------
    if phase == "countdown":
        elapsed = current_time - phase_start_time
        count = int(3 - elapsed)
        if count > 0:
            frame = put_text(frame,
                             f"[문제 {current_problem_idx + 1}/{len(PROBLEMS)}] 준비시간: {count}초\n정답: {PROBLEMS[current_problem_idx]}",
                             (50, 50), FONT_PATH, 32, (255, 255, 255))
        else:
            print(f"\n▼▼▼ 새 문제 시작: {PROBLEMS[current_problem_idx]} ▼▼▼")
            phase = "capture"
            captured_frames = []
            phase_start_time = current_time  # 🔥 캡처 시작 시간 초기화

    elif phase == "capture":
        frame = put_text(frame, "데이터 수집 중...", (50, 50),
                         FONT_PATH, font_size=32, color=(255, 255, 255))

        # 🔥 항상 프레임 데이터 저장 (손이 없어도 0으로 채움)
        landmarks = np.zeros(126)
        if results.multi_hand_landmarks:
            temp_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                temp_landmarks.extend([lm.x for lm in hand_landmarks.landmark])
                temp_landmarks.extend([lm.y for lm in hand_landmarks.landmark])
                temp_landmarks.extend([lm.z for lm in hand_landmarks.landmark])
            temp_landmarks = np.array(temp_landmarks[:126])
            landmarks[:len(temp_landmarks)] = temp_landmarks

        captured_frames.append(landmarks.tolist())
        print(f"→ 수집 프레임: {len(captured_frames)}/30", end='\r')  # 진행상황 표시

        # 🔥 30프레임 도달 체크 (조건문 위치 변경)
        if len(captured_frames) >= 30:
            print("\n✓ 30프레임 수집 완료 → 예측 수행")
            input_data = np.array(captured_frames).reshape(1, 30, 126)
            with torch.no_grad():
                output = model(torch.tensor(input_data, dtype=torch.float32))
                pred_class = torch.argmax(output).item()

            prediction = GESTURES.get(pred_class, "Unknown")
            total_results.append({
                "problem": PROBLEMS[current_problem_idx],
                "attempt": 3 - attempts_left + 1,
                "prediction": prediction
            })
            phase = "display"
            phase_start_time = current_time
            attempts_left -= 1

    elif phase == "display":
        current_result = total_results[-1]
        text = f"시도 {current_result['attempt']}번: {current_result['prediction']}"
        frame = put_text(frame, text, (50, 150), FONT_PATH, 40, (0, 0, 255))

        if current_time - phase_start_time >= 1:
            if attempts_left > 0:
                phase = "countdown"
                phase_start_time = current_time
            else:
                current_problem_idx += 1
                if current_problem_idx >= len(PROBLEMS):
                    break
                attempts_left = 3
                phase = "countdown"
                phase_start_time = current_time

    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 결과 리포트 출력
print("\n===== 최종 결과 =====")
for idx, problem in enumerate(PROBLEMS):
    problem_results = [r for r in total_results if r['problem'] == problem]
    correct_count = sum(1 for r in problem_results if r['prediction'] == problem)
    print(f"문제 {idx + 1}: {problem} | 시도횟수: {len(problem_results)} | 정답횟수: {correct_count}")

cap.release()
cv2.destroyAllWindows()
