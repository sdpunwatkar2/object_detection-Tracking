import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import pyttsx3
import datetime
import os

# Create required folders if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ========== Init ==========
model = YOLO("yolov8n.pt")  # Use yolov8n for speed, upgrade to yolov8m/l for accuracy
cap = cv2.VideoCapture(0)

save_path = "outputs"
os.makedirs(save_path, exist_ok=True)
log_path = os.path.join("logs", "detections.csv")
video_path = os.path.join(save_path, "annotated_output.avi")

# Voice Alert
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Create empty log file if not exists
if not os.path.exists(log_path):
    df = pd.DataFrame(columns=["Timestamp", "ObjectID", "Class", "EnteredZone"])
    df.to_csv(log_path, index=False)

# Writer for output video
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS) or 30
h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

# Define a polygon zone (sample diamond shape in center)
zone_pts = np.array([[(w//2, h//4), (3*w//4, h//2), (w//2, 3*h//4), (w//4, h//2)]], dtype=np.int32)
alerted_ids = set()
trajectories = {}  # ID: list of (x, y)

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection + Tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        xywhs = results[0].boxes.xywh.cpu().tolist()

        for obj_id, cls_id, xywh in zip(ids, classes, xywhs):
            cx, cy = int(xywh[0]), int(xywh[1])
            label = model.names[cls_id]

            # Draw trajectory
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append((cx, cy))
            for i in range(1, len(trajectories[obj_id])):
                cv2.line(annotated_frame, trajectories[obj_id][i-1], trajectories[obj_id][i], (0, 255, 255), 2)

            # Zone check
            if cv2.pointPolygonTest(zone_pts[0], (cx, cy), False) >= 0:
                if obj_id not in alerted_ids:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = pd.DataFrame([[timestamp, obj_id, label, "Yes"]],
                                             columns=["Timestamp", "ObjectID", "Class", "EnteredZone"])
                    log_entry.to_csv(log_path, mode='a', header=False, index=False)
                    speak(f"Alert! {label} entered the zone.")
                    alerted_ids.add(obj_id)

    # Draw zone
    cv2.polylines(annotated_frame, zone_pts, isClosed=True, color=(0, 0, 255), thickness=2)

    # Show and save
    cv2.imshow("AI Surveillance Pro+", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Surveillance ended. Log and video saved.")
