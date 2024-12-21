import cv2
import numpy as np

# Paths to model files
weights_path = r"C:\Users\gaura\Downloads\mobilenet_iter_73000.caffemodel"
config_path = r"C:\Users\gaura\Downloads\deploy.prototxt"
names_path = r"C:\Users\gaura\OneDrive\Documents\COCO.txt"

# Load class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the SSD model
net = cv2.dnn.readNet(weights_path, config_path)

# Initialize video capture
video_path = r"C:\Users\gaura\Downloads\854204-hd_1920_1080_30fps.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the video file.")
        break

    # Prepare the frame for SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Perform inference
    detections = net.forward()

    # Debugging: Check detection shape
    print("Detection shape:", detections.shape)

    # Parse detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:  # Lowered confidence threshold
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            )
            (x_start, y_start, x_end, y_end) = box.astype("int")

            # Draw bounding box
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

    # Show frame
    cv2.imshow("Object Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
