from PIL import Image
import numpy as np


def determine_flag(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((300, 200))  # Resize to a standard 3:2 aspect ratio
    data = np.array(image)

    height = data.shape[0]
    top_half = data[:height // 2, :, :]
    bottom_half = data[height // 2:, :, :]

    # Calculate average color
    def dominant_color(region):
        pixels = region.reshape(-1, 3)  # Flatten to list of RGB values
        mean_color = np.mean(pixels, axis=0)  # Mean color
        return mean_color

    top_color = dominant_color(top_half)
    bottom_color = dominant_color(bottom_half)

    # Expected colors
    red = np.array([200, 0, 0])
    white = np.array([255, 255, 255])
    tolerance = 70  # Adjusted tolerance

    def is_color_close(color1, color2, tol):
        return np.all(np.abs(color1 - color2) <= tol)

    if is_color_close(top_color, red, tolerance) and is_color_close(bottom_color, white, tolerance):
        return "Flag of Indonesia"
    elif is_color_close(top_color, white, tolerance) and is_color_close(bottom_color, red, tolerance):
        return "Flag of Poland"
    else:
        return "Neither Indonesia nor Poland flag"


# Example usage
image_path = r"C:\Users\gaura\Downloads\poland.png"  # Update this to your image path
result = determine_flag(image_path)
print(result)
