import cv2
import torch

# Load the YOLOv5 model (ensure you have yolov5 cloned and set up)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use a smaller model or your custom trained model

# Video input
video_path = r"C:\Users\gaura\Downloads\16714568-hd_1920_1080_60fps.mp4" # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened correctly
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Process detections
    for detection in results.xyxy[0]:  # detections for the first image
        x1, y1, x2, y2, conf, cls = detection  # bounding box coordinates, confidence, and class
        label = f"{model.names[int(cls)]}: {conf:.2f}"  # Get class name and confidence

        # Draw bounding box on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLOv5 Flag Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

