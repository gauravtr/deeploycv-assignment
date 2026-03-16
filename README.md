Projects Included
1. Real-Time Object Detection using YOLOv5

This project demonstrates how to perform object detection on video streams using the YOLOv5 deep learning model.

Features

Uses YOLOv5 pretrained model

Detects multiple objects in video frames

Displays bounding boxes, class labels, and confidence scores

Saves processed video with detections

Technologies Used

Python

PyTorch

OpenCV

YOLOv5

Example Workflow

Load YOLOv5 model using PyTorch Hub.

Read video frames using OpenCV.

Perform object detection on each frame.

Draw bounding boxes around detected objects.

Save the processed video output.

2. Hybrid Image Generation using Frequency Filters

This project generates hybrid images by combining high-frequency and low-frequency components from two different images.

Concept

Hybrid images appear different depending on viewing distance:

Close distance → high frequency image visible

Far distance → low frequency image visible

Implementation Steps

Convert images to grayscale

Apply Gaussian blur for low-pass filtering

Subtract blurred image to obtain high-pass filtered image

Combine both filtered images

Technologies Used

Python

OpenCV

NumPy

Matplotlib

3. Ridge Regression Model

This project implements Ridge Regression, a regularized linear regression technique that helps prevent overfitting.

Features

Interactive input of dataset

Accepts multiple features and samples

Adjustable regularization parameter (alpha)

Outputs regression coefficients and intercept

Technologies Used

Python

NumPy

Scikit-Learn
Installation

Clone the repository:

git clone https://github.com/your-username/computer-vision-ml-projects.git
cd computer-vision-ml-projects

Install required dependencies:

pip install numpy opencv-python torch scikit-learn matplotlib
Running the Projects
Run Ridge Regression
python ridge_regression.py
Run YOLOv5 Object Detection
python yolov5_detection.py
Run Hybrid Image Generation
python hybrid_image.py
Repository Structure
computer-vision-ml-projects
│
├── ridge_regression.py
├── yolov5_detection.py
├── hybrid_image.py
├── README.md
└── sample_videos_images
Learning Outcomes

Through these projects I explored:

Deep learning based object detection

Image processing using frequency filtering

Regularization techniques in machine learning models

Practical usage of OpenCV, PyTorch and Scikit-Learn

Future Improvements

Train a custom YOLO model on a specific dataset

Implement real-time webcam detection

Add model evaluation metrics

Create a simple web interface for object detection
