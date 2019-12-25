import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# scripts for assessing accuracy of subcellular localization
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/Contours/analyses/20190917_naming/'
# read in data
seg_data = xr.open_dataarray(base_dir + 'segmented_data/Point8/segmented_data.nc')


# create stacked barplots for subcellular localization of imaging signal
names = ["HH3", "Ecad", "LaminAC", "Phosphorous"]
positions = np.arange(len(names))

nuc_frac = np.zeros(len(names))
nuc_frac[0] = np.sum(seg_data.loc["nuc_mask", :, "HH3.tif"].values) / np.sum(seg_data.loc["cell_mask", :, "HH3.tif"])
nuc_frac[1] = np.sum(seg_data.loc["nuc_mask", :, "ECadherin.tif"].values) / np.sum(seg_data.loc["cell_mask", :, "ECadherin.tif"])
nuc_frac[2] = np.sum(seg_data.loc["nuc_mask", :, "LaminAC.tif"].values) / np.sum(seg_data.loc["cell_mask", :, "LaminAC.tif"])
nuc_frac[3] = np.sum(seg_data.loc["nuc_mask", :, "P.tif"].values) / np.sum(seg_data.loc["cell_mask", :, "P.tif"])

cell_frac = np.ones(len(names))
cell_frac = cell_frac - nuc_frac

plt.bar(x=positions, height=nuc_frac, width=1, edgecolor="white")
plt.bar(x=positions, height=cell_frac, bottom=nuc_frac, width=1, edgecolor="white")
plt.xticks(positions, names)
