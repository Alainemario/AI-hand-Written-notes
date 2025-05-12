import os
from PIL import Image
import cv2
import numpy as np

def optimize_image(input_path, output_path, max_width=1024, max_height=1024):
    """
    Optimize and compress scanned images to reduce size.
    - Converts to grayscale
    - Resizes image to a manageable size
    - Applies lossless compression
    """
    # Open image using Pillow
    img = Image.open(input_path)

    # Convert to grayscale
    img = img.convert("L")

    # Resize image if it exceeds max dimensions
    width, height = img.size
    if width > max_width or height > max_height:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            img = img.resize((max_width, int(max_width / aspect_ratio)))
        else:
            img = img.resize((int(max_height * aspect_ratio), max_height))
    
    # Save the optimized image
    img.save(output_path, quality=95, optimize=True)

def compress_images_in_directory(input_dir, output_dir, max_width=1024, max_height=1024):
    """
    Optimize and compress all images in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Optimizing: {input_path}")
            optimize_image(input_path, output_path, max_width, max_height)

if __name__ == "__main__":
    input_dir = "data/raw_notes"  # Path to your raw notes folder
    output_dir = "data/optimized_notes"  # Path where the compressed images will be saved
    compress_images_in_directory(input_dir, output_dir)
