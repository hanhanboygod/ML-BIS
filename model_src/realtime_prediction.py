import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
import sys
from pathlib import Path

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from absl import logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass

# reduce python logger for tensorflow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)



ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
# Assuming this import works in your environment
from MoveNet.realtime_detection import RealtimeDetector 

# --------- Settings ---------
MODEL_PATH = "best_model.keras"
CLASSES = ["cross", "uppercut", "rear_hook", "jab"]
SEQUENCE_LENGTH = 64
KEYPOINTS = 17
DIM = 2
CHANNELS = 1

# --------- Skeleton Connections (Draw lines) ---------
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', 
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm', 
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm', 
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', 
    (12, 14): 'c', (14, 16): 'c'
}

# --------- Load Model ---------
print("Loading Keras model...")
try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded. Input shape:", getattr(model, "input_shape", None), flush=True)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --------- Initialize MoveNet ---------
print("Initializing MoveNet...")
detector = RealtimeDetector(
    iou_threshold=0.3,
    detection_threshold=0.5,
    min_keypoints=6,
    verbose=False
)

# --------- Helper Functions ---------
def movenet_detect(frame):
    """ 
    Returns 17x2 np.array in RAW MoveNet format [y, x].
    We do NOT swap here, because the Model likely expects [y, x].
    """
    detections = detector.detect_people(frame)
    if len(detections) == 0:
        return np.zeros((KEYPOINTS, DIM), dtype=np.float32)
    
    bbox, kp = detections[0]
    # kp is [y, x, score]. Return [y, x] for the model.
    return kp[:, :2]

def draw_skeleton(frame, keypoints_yx):
    """ 
    Takes keypoints in [y, x] format and swaps to [x, y] for drawing.
    """
    h, w, _ = frame.shape
    
    # Convert [y, x] to [x, y] just for visualization
    keypoints_xy = np.stack([keypoints_yx[:, 1], keypoints_yx[:, 0]], axis=1)
    
    # Draw Lines (Limbs)
    for (p1, p2), color_code in EDGES.items():
        x1, y1 = keypoints_xy[p1]
        x2, y2 = keypoints_xy[p2]
        
        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    # Draw Points (Joints)
    for (kx, ky) in keypoints_xy:
        if kx > 0 and ky > 0:
            px, py = int(kx * w), int(ky * h)
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

# --------- Main ---------
def main(video_path_str):
    print(f"Attempting to open: {video_path_str}", flush=True)

    try:
        video_source = int(video_path_str) if str(video_path_str).isdigit() else video_path_str
    except:
        video_source = video_path_str

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("ERROR: Could not open video source. Check path.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video Opened: {w}x{h} @ {fps}fps. Total Metadata Frames: {total_frames}")

    cv2.namedWindow("Real-Time Boxing Action", cv2.WINDOW_NORMAL)
    
    sequence_buffer = []
    current_label = "Waiting..."
    current_score = 0.0
    frames_processed = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "output_annotated.mp4"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # --- MAIN LOOP ---
    while True:
        ret, frame = cap.read()
        
        # If video ends or read fails
        if not ret:
            # CHECK: If we processed frames but haven't filled buffer (Video Too Short)
            if frames_processed > 0 and len(sequence_buffer) < SEQUENCE_LENGTH:
                print(f"\nVideo ended early ({frames_processed} frames). Padding buffer to force prediction...")
                
                # Get the last valid keypoints (or zeros if none)
                last_kp = sequence_buffer[-1] if sequence_buffer else np.zeros((KEYPOINTS, DIM), dtype=np.float32)
                
                # Create a black frame (or use last frame) for padding visualization
                pad_frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                # PAD LOOP: Fill the rest of the buffer
                while len(sequence_buffer) < SEQUENCE_LENGTH:
                    sequence_buffer.append(last_kp)
                    
                    # Run UI logic for padding frames so they appear in video
                    draw_skeleton(pad_frame, last_kp)
                    
                    # (Prediction logic copy-paste for padding phase)
                    if len(sequence_buffer) == SEQUENCE_LENGTH:
                        input_seq = np.array(sequence_buffer, dtype=np.float32)
                        input_seq = input_seq[np.newaxis, ..., np.newaxis]
                        try:
                            pred = model.predict(input_seq, verbose=0)[0]
                            idx = int(np.argmax(pred))
                            current_score = float(pred[idx])
                            current_label = CLASSES[idx] if current_score > 0.4 else "Uncertain"
                            print(f"PADDED PREDICTION: {current_label} ({current_score:.2f})")
                        except Exception as e:
                            print(e)

                    # Draw text on pad frame
                    cv2.rectangle(pad_frame, (0, 0), (350, 60), (0, 0, 0), -1)
                    cv2.putText(pad_frame, f"PADDING: {current_label} ({current_score:.2f})", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    if writer.isOpened():
                        writer.write(pad_frame)
                
                # Break after padding
                break
            else:
                print("\nEnd of video stream.")
                break

        frames_processed += 1
        print(f"Processing frame {frames_processed}...", end='\r')

        # 1. Detection [y, x]
        keypoints_yx = movenet_detect(frame)
        
        # 2. Buffer
        sequence_buffer.append(keypoints_yx)
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # 3. Draw [x, y]
        draw_skeleton(frame, keypoints_yx)

        # 4. Predict
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            input_seq = np.array(sequence_buffer, dtype=np.float32)
            input_seq = input_seq[np.newaxis, ..., np.newaxis] 
            
            try:
                pred = model.predict(input_seq, verbose=0)[0]
                idx = int(np.argmax(pred))
                current_score = float(pred[idx])
                
                if current_score > 0.4:
                    current_label = CLASSES[idx]
                else:
                    current_label = "Uncertain"
            except Exception as e:
                pass

        # 5. UI
        bar_width = int((len(sequence_buffer) / SEQUENCE_LENGTH) * w)
        cv2.rectangle(frame, (0, h-10), (bar_width, h), (0, 255, 0), -1)
        
        cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
        status_color = (0, 255, 0)
        if current_label == "Waiting...": status_color = (0, 255, 255)
        elif current_label == "Uncertain": status_color = (0, 0, 255)
            
        text = f"{current_label} ({current_score:.2f})"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Show & Write
        try:
            cv2.imshow("Real-Time Boxing Action", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass

        if writer.isOpened():
            writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frames_processed} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Video path")
    args = parser.parse_args()
    main(args.video)