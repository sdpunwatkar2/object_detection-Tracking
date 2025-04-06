import cv2
import os
import json
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Initialize folders
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs/snapshots", exist_ok=True)
os.makedirs("config", exist_ok=True)

# Load detection model
model = YOLO("yolov8n.pt")

# Filter to detect only people
CLASSES = ["person"]

# Load zones from config
with open("config/zones.json") as f:
    zones = json.load(f)

# Initialize database
conn = sqlite3.connect("logs/detections.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS logs (
                timestamp TEXT, 
                zone TEXT, 
                speed REAL,
                snapshot_path TEXT
            )''')
conn.commit()

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Track object IDs and previous positions
object_tracker = {}
snapshot_counter = 0

print("[INFO] Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    annotated_frame = results[0].plot()

    for r in results:
        if r.boxes.id is None:
            continue

        for box, cls_id, track_id in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):
            class_name = model.names[int(cls_id)]
            if class_name != "person":
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_id = int(track_id.item())

            # Check zones
            for zone in zones:
                zx1, zy1, zx2, zy2 = zone["coords"]
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    zone_name = zone["name"]

                    # Speed Estimation
                    speed = 0.0
                    if track_id in object_tracker:
                        prev_cx, prev_cy, prev_time = object_tracker[track_id]
                        dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                        time_elapsed = (datetime.now() - prev_time).total_seconds()
                        speed = (dist / time_elapsed) * (fps / 100)  # simplistic unit

                    object_tracker[track_id] = (cx, cy, datetime.now())

                    # Snapshot if in Restricted Zone
                    snap_path = ""
                    if zone_name == "Restricted":
                        snapshot_counter += 1
                        snap_path = f"outputs/snapshots/person_{snapshot_counter}.jpg"
                        cropped = frame[y1:y2, x1:x2]
                        cv2.imwrite(snap_path, cropped)

                    # Log to SQLite
                    c.execute("INSERT INTO logs (timestamp, zone, speed, snapshot_path) VALUES (?, ?, ?, ?)",
                              (now, zone_name, round(speed, 2), snap_path))
                    conn.commit()

                    # Voice alert for restricted
                    if zone_name == "Restricted":
                        print("[ALERT] Person entered restricted zone!")

    # Draw zones
    for zone in zones:
        zx1, zy1, zx2, zy2 = zone["coords"]
        cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), zone["color"], 2)
        cv2.putText(annotated_frame, zone["name"], (zx1, zy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone["color"], 2)

    cv2.imshow("AI Surveillance Pro+", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()