# take segmented data and specific input channels and combine into xarray for caliban
import xarray as xr
import numpy as np
from segmentation import helper_functions

base_dir = "/Users/noahgreenwald/Documents/MIBI_Data/selena/20190925_PAH_project/PAHTrainingData/"

segmentation_xr = xr.open_dataarray(base_dir + "/segmentation_output/segmentation_labels.nc")

channel_xr = xr.open_dataarray(base_dir + "no_noise/Deepcell_Input_Na.nc")
channel_xr = helper_functions.load_tifs_from_points_dir(base_dir + "Na_no_background", tif_folder="TIFs",
                                                         points=segmentation_xr.points.values, tifs=["Na.tif", "H3.tif"])

channel_xr1 = helper_functions.load_tifs_from_points_dir(base_dir + "python_denoised", tif_folder="",
                                                         points=segmentation_xr.points.values, tifs=["Na_rescaled.tif"])


# create new numpy array to hold selected channel data and segmentation labels
caliban_np = np.zeros((channel_xr.shape[:-1] + (channel_xr.shape[-1] + 2, )))

caliban_np[:, :, :, :-2] = channel_xr.values
caliban_np[:, :, :, -2:-1] = channel_xr1.values

caliban_np[:, :, :, -1:] = segmentation_xr.values

channel_names = np.append(np.append(channel_xr.channels.values, channel_xr1.channels.values), segmentation_xr.channels.values)

caliban_xr = xr.DataArray(caliban_np, coords=[channel_xr.points, channel_xr.rows, channel_xr.cols, channel_names],
                          dims=["points", "rows", "cols", "channels"])

caliban_xr.to_netcdf(base_dir + "caliban_input/caliban_input.nc", format="NETCDF3_64BIT")