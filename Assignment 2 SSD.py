import cv2
import numpy as np


weights_path = r"C:\Users\gaura\Downloads\mobilenet_iter_73000.caffemodel"
config_path = r"C:\Users\gaura\Downloads\deploy.prototxt"
names_path = r"C:\Users\gaura\OneDrive\Documents\COCO.txt"


with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]


net = cv2.dnn.readNet(weights_path, config_path)


video_path = r"C:\Users\gaura\Downloads\854204-hd_1920_1080_30fps.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the video file.")
        break


    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)


    detections = net.forward()


    print("Detection shape:", detections.shape)


    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:  
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            )
            (x_start, y_start, x_end, y_end) = box.astype("int")

     
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )


    cv2.imshow("Object Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

