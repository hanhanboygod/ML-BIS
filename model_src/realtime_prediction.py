import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from MoveNet.realtime_detection import RealtimeDetector  # MoveNet MultiPose 偵測器

# --------- 設定 ---------
MODEL_PATH = "best_model.keras"
CLASSES = ["cross", "uppercut", "rear_hook", "jab"]
SEQUENCE_LENGTH = 64   # CNN-LSTM 訓練時的 sequence 長度
KEYPOINTS = 17
DIM = 2  # x,y
CHANNELS = 1

# --------- 載入 CNN-LSTM 模型 ---------
model = load_model(MODEL_PATH)

# --------- 初始化 MoveNet偵測器 ---------
detector = RealtimeDetector(
    iou_threshold=0.3,
    detection_threshold=0.5,
    min_keypoints=6,
    verbose=False
)

# --------- frame buffer ---------
sequence_buffer = []

# --------- 偵測 keypoints ---------
def movenet_detect(frame):
    """
    輸入: BGR frame
    輸出: 17x2 np.array, normalized [0,1]
    """
    detections = detector.detect_people(frame)
    if len(detections) == 0:
        return np.zeros((KEYPOINTS, DIM), dtype=np.float32)
    
    # 取第一個人
    bbox, kp = detections[0]
    keypoints = kp[:, :2]  # 只要 x,y
    return keypoints

# --------- 偵測人物框 ---------
def detect_person_bbox(keypoints, frame_shape):
    """
    keypoints: (17,2) normalized
    frame_shape: (height, width, channels)
    """
    h, w, _ = frame_shape
    x = keypoints[:,0] * w
    y = keypoints[:,1] * h
    x1, y1 = int(np.min(x)), int(np.min(y))
    x2, y2 = int(np.max(x)), int(np.max(y))
    return x1, y1, x2, y2

# --------- 主程式 ---------
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    global sequence_buffer

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 偵測 keypoints
        keypoints = movenet_detect(frame)
        sequence_buffer.append(keypoints)

        # 保持序列長度
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # 當序列滿了，進行預測
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            input_seq = np.array(sequence_buffer, dtype=np.float32)  # (64,17,2)
            input_seq = input_seq[np.newaxis, ..., np.newaxis]       # (1,64,17,2,1)
            pred = model.predict(input_seq, verbose=0)[0]            # softmax
            predicted_class = np.argmax(pred)
            score = pred[predicted_class]

            # 畫框
            x1, y1, x2, y2 = detect_person_bbox(keypoints, frame.shape)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            text = f"{CLASSES[predicted_class]}: {score:.4f}"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Real-Time Boxing Action", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------- CLI ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Boxing Action Prediction")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    main(args.video)
