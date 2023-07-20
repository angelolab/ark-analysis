import os
from PIL import Image
import numpy as np


def renumber_masks(root_dir):
    # Initialize a global label counter
    global_label_counter = 0

    # Loop through all subdirectories
    for subdir, dirs, files in os.walk(root_dir):
        # Loop through all files in the current subdirectory
        for file in files:
            # Check if the file is a tiff image (you can modify this to match your mask file type)
            if file.endswith('.tiff'):
                # Construct the full file path
                file_path = os.path.join(subdir, file)

                # Load the image file as a numpy array
                img = np.array(Image.open(file_path))

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
                Image.fromarray(img).save(file_path)
