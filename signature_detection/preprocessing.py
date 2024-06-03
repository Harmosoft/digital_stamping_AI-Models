import os
import cv2
from tqdm import tqdm

# Define paths to input and output directories
input_dir = 'data/raw_images'
output_dir = 'data/processed_images'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to preprocess images
def preprocess_images(input_dir, output_dir):
    # Get list of image file names in input directory
    image_files = os.listdir(input_dir)

    # Loop through each image file
    for image_file in tqdm(image_files):
        # Read image
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        # Resize image to desired dimensions (e.g., 224x224 pixels)
        resized_image = cv2.resize(image, (224, 224))

        # Convert image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to smooth image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply adaptive thresholding to binarize image
        _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save preprocessed image to output directory
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, thresholded_image)

# Preprocess images in input directory and save to output directory
preprocess_images(input_dir, output_dir)
