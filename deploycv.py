import numpy as np
import cv2
from matplotlib import pyplot as plt

def apply_high_pass_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (21, 21), 0)
    high_pass = cv2.subtract(gray_image, blurred)
    return high_pass

def apply_low_pass_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low_pass = cv2.GaussianBlur(gray_image, (21, 21), 0)
    return low_pass

def resize_image(image, target_shape):
    resized = cv2.resize(image, (target_shape[1], target_shape[0]))
    return resized

def combine_images(high_pass_image, low_pass_image):
    if high_pass_image.shape != low_pass_image.shape:
        low_pass_image = resize_image(low_pass_image, high_pass_image.shape)
    combined = cv2.add(high_pass_image, low_pass_image)
    return combined

def show(name, n, m, i, Title):
    plt.subplot(n, m, i)
    plt.imshow(name, cmap='gray' if len(name.shape) == 2 else None)
    plt.title(Title)
    plt.axis('off')

image1_path = r"C:\Users\gaura\OneDrive\Desktop\UPSC\LAW\DESIEL.jpeg"
image2_path = r"C:\Users\gaura\OneDrive\Desktop\UPSC\LAW\YOGI.jpeg"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

target_shape = image1.shape[:2]
image2_resized = resize_image(image2, target_shape)

high_pass_image = apply_high_pass_filter(image1)
low_pass_image = apply_low_pass_filter(image2_resized)

combined_image = combine_images(high_pass_image, low_pass_image)

plt.figure(figsize=(12, 8))
show(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), 2, 3, 1, "Original Image 1")
show(cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB), 2, 3, 2, "Original Image 2 (Resized)")
show(high_pass_image, 2, 3, 3, "High-Pass Filtered")
show(low_pass_image, 2, 3, 4, "Low-Pass Filtered")
show(combined_image, 2, 3, 5, "Combined Image")

plt.tight_layout()
plt.show()
