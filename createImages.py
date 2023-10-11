import os
import pandas as pd
import numpy as np
import cv2
from scipy import interpolate
from PIL import Image

# Define the original shape (16x1) and the desired output shape (64x64)
original_shape = (16, 1)
output_shape = (64, 64)

# Create a function to resize the array using bilinear interpolation
def resize_array(input_array, output_size):
    x = np.arange(0, input_array.shape[1])
    y = np.arange(0, input_array.shape[0])
    
    # Create a meshgrid for the original array's coordinates
    x_orig, y_orig = np.meshgrid(x, y)
    
    # Create a meshgrid for the resized array's coordinates
    x_new = np.linspace(0, input_array.shape[1] - 1, output_size[1])
    y_new = np.linspace(0, input_array.shape[0] - 1, output_size[0])
    x_new, y_new = np.meshgrid(x_new, y_new)
    
    # Use bilinear interpolation to resize the array
    interpolated_array = interpolate.griddata(
        (x_orig.flatten(), y_orig.flatten()), input_array.flatten(),
        (x_new, y_new), method='linear', fill_value=0
    )
    
    return interpolated_array


# Define the path to your CSV file
csv_file = 'final_train_data.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file)

# Create the output folders if they don't exist
output_folders = ['anomaly', 'normal']
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    # Extract the class and feature data
    class_name = row['class']
    feature_data = row.drop('class').values.astype(np.float32)
    
    print(feature_data)
    
    reshaped_64x64_array = np.tile(feature_data, (64, 64))

    # Reshape the feature data to 64x64 (assuming it can be reshaped)
    image_data = reshaped_64x64_array

    # Normalize the image data to 0-255 range
    image_data = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)

    # Create a PIL Image from the normalized image data
    image = Image.fromarray(image_data)

    # Define the output path based on the class
    output_folder = 'anomaly' if class_name == 'anomaly' else 'normal'
    output_path = os.path.join(output_folder, f'image_{index}.png')

    # Save the image as a PNG file
    image.save(output_path)

    print(f'Saved image {index} to {output_path}')
