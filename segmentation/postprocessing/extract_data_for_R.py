import numpy as np
import pandas as pd
import os
from skimage.measure import regionprops
import math
import skimage.io as io
base_dir = "/Users/noahgreenwald/Documents/MIBI_Data/felix/20191001_cohort/fcs_3_pixel_3smooth/"

files = os.listdir(base_dir)
files = [file for file in files if "Label_Map" in file]

region_data = pd.DataFrame(columns=["point", "cell_id", "center_rowcoord", "center_colcoord", "cell_size"])

for file in files:
    # read in label map for each point
    print("analyszing file {}".format(file))
    label_map = io.imread(base_dir + file)
    props = regionprops(label_map)

    for i in range(1, len(props)):
        coords = props[i].centroid
        size = props[i].area
        cell_id = props[i].label
        region_data = region_data.append({"point": file, "cell_id": cell_id,
                            "center_rowcoord": math.floor(coords[0]), "center_colcoord": math.floor(coords[1]),
                            "cell_size": size}, ignore_index=True)


region_data.to_csv(base_dir + "combined_spatial.csv")