import numpy as np
import PIL
from PIL import Image

import os
def is_mostly_white(image, threshold=0.97, color=0.03, white=True):
    # Open the image and convert it to grayscale
    
    image_array = np.array(image)

    # Normalize pixel values to range 0-1 (0 = black, 1 = white)
    normalized_pixels = image_array / 255.0

    pixel_count = 0

    # Count white or pixels (close to 1.0)
    if not white:
        pixel_count = np.sum(normalized_pixels < 0.03)  # Adjust tolerance if needed
    else:
        pixel_count = np.sum(normalized_pixels > 0.97)
    
    total_pixel_count = normalized_pixels.size

    # Calculate the percentage of white pixels
    pixel_percentage = pixel_count / total_pixel_count
    print(pixel_percentage)

    # Return True if white pixels exceed the threshold
    return pixel_percentage > threshold

mem_folder = "memorized_images"
for i in range(500):
    imgname = "{}.jpg".format(str(i))
    imname = "{}.png".format(str(i))
    print(imname)
    memorized_path_background = os.path.join(mem_folder, "background", imname)
    memorized_path_foreground = os.path.join(mem_folder, "foreground", imname) 
    mem_path = os.path.join(mem_folder, imgname) 

    try:
        mem_background = Image.open(memorized_path_background).convert('RGBA')
        mem_foreground = Image.open(memorized_path_foreground).convert('RGBA')
        mem_image = Image.open(mem_path).convert('RGBA')
    except FileNotFoundError:
        continue

    mem_background_copy = mem_background.convert("L")
    mem_foreground_copy = mem_foreground.convert("L")
    if is_mostly_white(mem_background_copy):
        mem_image.save(memorized_path_background)
        print("Using Original Image as Background for Image {i}")
    if is_mostly_white(mem_foreground_copy, white=False):
        mem_image.save(memorized_path_foreground)
        print("Using Original Image as Foreground for Image {i}")
                 
