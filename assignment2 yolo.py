import cv2
import torch
import numpy as np
from pathlib import Path

# Load YOLOv5 model
model_path = "yolov5s.pt"  # Path to the YOLOv5 model file (small model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Load custom model or pretrained yolov5s
model.conf = 0.3  # Confidence threshold

# Video input
video_path = r"C:\Users\gaura\Downloads\854204-hd_1920_1080_30fps.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened correctly
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get video frame dimensions and initialize output writer
output_path = "output_yolov5.avi"
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (YOLOv5 expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(rgb_frame)

    # Extract detections
    detections = results.xyxy[0].cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2, confidence, class)

    # Draw detections on the frame
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        color = (0, 255, 0)  # Green for bounding boxes
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("YOLOv5 Object Detection", frame)

    # Write the frame to the output file
    output.write(frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_path}")
