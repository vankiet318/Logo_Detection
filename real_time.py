import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("./static/models/yolo/best.pt")  # Update with your actual model path

# Start webcam
cap = cv2.VideoCapture(0)

# Check if camera is available
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run YOLO detection on the frame
    results = model(frame)

    # Copy the frame to apply blurring
    blurred_frame = frame.copy()

    # Get detection boxes
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for box in boxes:
        x1, y1, x2, y2 = box
        # Ensure box is within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # Extract and blur the detected region
        roi = blurred_frame[y1:y2, x1:x2]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (101, 101), 0)
            blurred_frame[y1:y2, x1:x2] = blurred_roi

            # Optionally draw the box (green)
            cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the live video feed
    cv2.imshow("🟢 Real-Time Logo Detection & Blurring", blurred_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
