from ultralytics import YOLO
import cv2

# Load YOLOv8 model (you can use yolov8n.pt for speed or yolov8m/l for accuracy)
model = YOLO("yolov8n.pt")

# Open webcam (0) or change to video file
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Draw results
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection + Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
