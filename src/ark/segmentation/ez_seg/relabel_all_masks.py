from typing import Generator
from skimage.io import imread
from alpineer import io_utils
from alpineer.image_utils import save_image
import numpy as np

import pathlib

def renumber_masks(mask_dir):
    mask_dir = pathlib.Path(mask_dir)
    io_utils.validate_paths(mask_dir)
    
    all_images: Generator[pathlib.Path, None, None] = mask_dir.rglob("*.tiff")

    global_unique_labels = 1

    for image in all_images:
        img: np.ndarray = imread(image)
        unique_labels: np.ndarray = np.unique(img)
        for label in unique_labels:
            if label != 0:
                global_unique_labels += 1

    all_images: Generator[pathlib.Path, None, None] = mask_dir.rglob("*.tiff")

    for image in all_images:
        print("Relabeling: " + image.stem + image.suffix)
        img: np.ndarray = imread(image)
        unique_labels: np.ndarray = np.unique(img)
        for label in unique_labels:
            if label != 0:
                img[img == label] = global_unique_labels
                global_unique_labels += 1
        save_image(fname=image, data=img)
    print("Complete.")
    
    
    
    