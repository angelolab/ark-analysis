import os
from skimage.io import imread
from alpineer.image_utils import save_image
import numpy as np


def renumber_masks(root_dir):
    # Initialize a global label counter
    global_label_counter = 0

    # Loop through all subdirectories
    for subdir, _, files in os.walk(root_dir):
        # Loop through all files in the current subdirectory
        for file in files:
            if file.endswith('.tiff'):
                # Construct the full file path
                file_path = os.path.join(subdir, file)

                # Load the image file as a numpy array
                img = imread(file_path)

                # Find all unique labels in the image
                unique_labels = np.unique(img)

                # Loop through all unique labels
                for label in unique_labels:
                    # Check if the label is not background (assuming background is labeled as 0)
                    if label != 0:
                        # Remap the label
                        img[img == label] = global_label_counter
                        # Increment the global label counter
                        global_label_counter += 1

                # Save the remapped image back to the file
                save_image(fname=file_path, data=img)
