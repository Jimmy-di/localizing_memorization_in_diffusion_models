from PIL import Image
import numpy as np

def is_mostly_white(image_path, threshold=0.97):
    # Open the image and convert it to grayscale
    
    image_array = np.array(image)

    # Normalize pixel values to range 0-1 (0 = black, 1 = white)
    normalized_pixels = image_array / 255.0

    # Count white pixels (close to 1.0)
    white_pixel_count = np.sum(normalized_pixels < 0.02)  # Adjust tolerance if needed
    total_pixel_count = normalized_pixels.size

    # Calculate the percentage of white pixels
    white_pixel_percentage = white_pixel_count / total_pixel_count
    print(white_pixel_percentage)

    # Return True if white pixels exceed the threshold
    return white_pixel_percentage > threshold

# Example usage
image_path = "generated_images_additional_prompts_500_1_foreground/img_0029_01.jpg"
image = Image.open(image_path).convert('L')  # 'L' mode converts to grayscale
if is_mostly_white(image_path):
    print("The image is over 95% white pixels.")
else:
    print("The image is not over 95% white pixels.")