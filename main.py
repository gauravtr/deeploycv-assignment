import cv2
import numpy as np
from matplotlib import pyplot as plt

def capture_image():
    cap = cv2.VideoCapture(0)  # Change to 1 if needed
    ret, frame = cap.read()
    cap.release()
    return frame

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold_image(image):
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return thresh

def two_color_image(image):
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return binary

def sixteen_gray_colors(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quantized = (gray_image // 16) * 16
    return quantized

def sobel_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return sobel_combined

def canny_edge_detector(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

def gaussian_blur(image):
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]]) / 256
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Capture an image
image = capture_image()

# Process the image
gray_image = to_gray(image)
thresholded_image = threshold_image(gray_image)
two_color_img = two_color_image(gray_image)
sixteen_gray_image = sixteen_gray_colors(image)
sobel_image = sobel_filter(image)
canny_edges = canny_edge_detector(image)
blurred_image = gaussian_blur(image)
sharpened_image = sharpen_image(blurred_image)

# Plotting all images in a 2x4 grid
plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(thresholded_image, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(two_color_img, cmap='gray')
plt.title("Two Color Image")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(sixteen_gray_image, cmap='gray')
plt.title("16 Gray Colors Image")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(sobel_image, cmap='gray')
plt.title("Sobel Filter")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(canny_edges, cmap='gray')
plt.title("Canny Edge Detector")
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(sharpened_image)
plt.title("Sharpened Image")
plt.axis('off')

plt.tight_layout()
plt.show()
