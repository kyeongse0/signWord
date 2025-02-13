import cv2
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 수어 단어 목록 (유의어 그룹)
GESTURES = [
    '안녕,안부',
    '실수',
    '살다,삶,생활',
    '가족,식구,세대,가구',
    '취미',
    '아빠,부,부친,아비,아버지',
    '건강,기력,강건하다,튼튼하다',
    '꿈,포부,꿈꾸다',
    '병원,의원',
    '어머니,모친,어미,엄마',
    '노래,음악,가요',
    '검사',
    '쉬다,휴가,휴게,휴식,휴양',
    '바쁘다,분주하다',
    '여행',
    '주무시다,자다,잠들다,잠자다',
    '고모',
    '치료',
    '자유,임의,마구,마음껏,마음대로,멋대로,제멋대로,함부로',
    '다니다',
    '이기다,승리,승리하다,(경쟁 상대를) 제치다',
    '낫다,치유',
    '성공',
    '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종',
    '오빠,오라버니',
    '누나,누님',
    '머무르다,존재,체류,계시다,묵다',
    '형,형님',
    '남편,배우자,서방',
    '축구,차다',
    '실패',
    '입원',
    '양치질,양치',
    '아들',
    '형제',
    '몸살',
    '약',
    '결혼,혼인,화혼',
    '남동생',
    '감기',
    '안과',
    '습관,버릇',
    '상하다,다치다,부상,상처,손상',
    '수술',
    '동생',
    '모자(관계)',
    '시동생',
    '편찮다,아프다',
    '신기록',
    '할머니,조모'
]

# 설정값
DATA_PATH = 'sign_language_data'  # 데이터 저장 경로
FRAME_COUNT = 30  # 프레임 개수 (1 시퀀스 당 30개 프레임)
SEQUENCES = 30  # 시퀀스 개수 (1 제스처 당 30개의 시퀀스)
FILES_PER_GESTURE = 40  # 일반 단어 파일 개수
SPECIAL_FILES_PER_GESTURE = 150  # 특별 단어 파일 개수

USER_NAME = "목경서"  # 사용자 이름 설정

# 특별히 더 많은 데이터가 필요한 단어들
SPECIAL_GESTURES = {
    # 예시로 몇 개 추가
}

# 폰트 설정
try:
    FONT_PATH = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"
    font = ImageFont.truetype(FONT_PATH, 32)
except OSError:
    try:
        FONT_PATH = "C:/Windows/Fonts/gulim.ttc"
        font = ImageFont.truetype(FONT_PATH, 32)
    except OSError:
        font = ImageFont.load_default()
        print("Warning: 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")


class SignLanguageDataCollector:
    def __init__(self):
        self.setup_directories()
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        self.font = font

    def setup_directories(self):
        """데이터 저장을 위한 디렉토리 생성"""
        os.makedirs(DATA_PATH, exist_ok=True)
        for gesture in GESTURES:
            os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

    def process_frame(self, frame):
        """프레임 처리 및 랜드마크 추출"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        '''
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
             cv2.putText(frame, 'Need both hands! Detected: ' + 
                     str(len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0), 
                     (10, frame.shape[0] - 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             return frame, None
          '''

        if not results.multi_hand_landmarks:
            cv2.putText(frame, 'No hands detected!',
                        (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, None

        # 항상 126개의 값을 가지는 배열 생성 (21개 랜드마크 * 3좌표 * 2손)
        landmarks = np.zeros(126)
        temp_landmarks = []

        # 랜드마크 추출
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks,
                                                               results.multi_handedness)):
            hand_label = handedness.classification[0].label
            cv2.putText(frame, f'Hand {idx + 1}: {hand_label}',
                        (10, frame.shape[0] - (60 + 30 * idx)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            temp_landmarks.extend([lm.x for lm in hand_landmarks.landmark])
            temp_landmarks.extend([lm.y for lm in hand_landmarks.landmark])
            temp_landmarks.extend([lm.z for lm in hand_landmarks.landmark])

        temp_landmarks = np.array(temp_landmarks[:126])
        landmarks[:len(temp_landmarks)] = temp_landmarks

        cv2.putText(frame, 'Both hands detected!',
                    (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, landmarks

    def draw_info(self, frame, gesture, sequence, total_sequences, frame_num):
        """프레임에 정보 표시"""
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        gesture_for_filename = gesture  # 전체 단어 그룹을 파일명으로 사용

        text = [
            f"의미 그룹: {gesture}",
            f"파일명: {gesture_for_filename}_{USER_NAME}_{sequence + 1}",
            f"시퀀스: {sequence + 1}/{total_sequences}",
            f"프레임: {frame_num + 1}/{FRAME_COUNT}"
        ]

        y = 10
        for line in text:
            draw.text((10, y), line, font=self.font, fill=(255, 255, 255))
            y += 40

        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    def collect_sequence_data(self, cap, gesture, sequence, total_sequences):
        """한 시퀀스의 데이터 수집"""
        frames_data = []
        retry_count = 0
        max_retries = 5  # 최대 재시도 횟수

        # 준비 시간 카운트다운
        for i in range(2, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return None

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            text = f"준비: {i}"
            draw.text((frame.shape[1] // 2 - 50, frame.shape[0] // 2), text,
                      font=self.font, fill=(255, 255, 255))
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1000)

        # 프레임 수집
        frame_num = 0
        while frame_num < FRAME_COUNT:
            ret, frame = cap.read()
            if not ret:
                return None

            frame, landmarks = self.process_frame(frame)

            if landmarks is None:  # 두 손이 모두 인식되지 않은 경우
                retry_count += 1
                if retry_count > max_retries:
                    print(f"\n두 손이 {max_retries}회 이상 인식되지 않아 현재 시퀀스를 다시 시작합니다.")
                    return None
                continue

            frames_data.append(landmarks)
            frame = self.draw_info(frame, gesture, sequence, total_sequences, frame_num)
            cv2.imshow('Data Collection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

            frame_num += 1
            retry_count = 0  # 성공적으로 프레임을 수집하면 재시도 카운트 초기화

        if len(frames_data) == FRAME_COUNT:
            return np.array(frames_data)
        return None

    def collect_data(self, gesture_index=None):
        """데이터 수집 실행"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            if gesture_index is not None:
                gestures_to_collect = [GESTURES[gesture_index]]
            else:
                gestures_to_collect = GESTURES

            for gesture in gestures_to_collect:
                gesture_for_path = gesture  # 전체 단어 그룹을 경로명으로 사용
                gesture_path = os.path.join(DATA_PATH, gesture_for_path)

                # 특별 단어인 경우 더 많은 데이터 수집
                if gesture_for_path in SPECIAL_GESTURES:
                    target_sequences = SPECIAL_FILES_PER_GESTURE
                else:
                    target_sequences = FILES_PER_GESTURE

                # 이미 존재하는 파일 수 확인
                existing_files = len([f for f in os.listdir(gesture_path)
                                      if f.endswith('.npy')])

                # 남은 시퀀스만큼만 수집
                for sequence in range(existing_files, target_sequences):
                    print(f"\n{gesture_for_path} - 시퀀스 {sequence + 1}/{target_sequences} 수집 중...")

                    while True:
                        frames_data = self.collect_sequence_data(cap, gesture, sequence, target_sequences)
                        if frames_data is None:
                            user_input = input("\n데이터 수집에 실패했습니다. 다시 시도하시겠습니까? (y/n): ")
                            if user_input.lower() != 'y':
                                raise KeyboardInterrupt
                            print("\n다시 시도합니다...")
                            continue

                        # 데이터 저장
                        filename = f"{gesture_for_path}_{USER_NAME}_{sequence + 1}.npy"
                        save_path = os.path.join(gesture_path, filename)
                        np.save(save_path, frames_data)
                        print(f"{filename} 저장완료")
                        break

        except KeyboardInterrupt:
            print("\n데이터 수집 중단됨")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


def print_gesture_list():
    """제스처 목록 출력"""
    print("\n=== 제스처 목록 ===")
    for idx, gesture in enumerate(GESTURES):
        print(f"[{idx}] {gesture}")
    print("==================\n")


if __name__ == "__main__":
    collector = SignLanguageDataCollector()

    while True:
        print_gesture_list()
        try:
            user_input = input("수집할 제스처의 인덱스를 입력하세요 (전체 수집: a, 종료: q): ")

            if user_input.lower() == 'q':
                print("프로그램을 종료합니다.")
                break
            elif user_input.lower() == 'a':
                collector.collect_data()
                break
            else:
                index = int(user_input)
                if 0 <= index < len(GESTURES):
                    print(f"\n'{GESTURES[index]}' 데이터 수집을 시작합니다...")
                    collector.collect_data(index)
                    break
                else:
                    print("올바른 인덱스를 입력해주세요.")
        except ValueError:
            print("올바른 값을 입력해주세요.")