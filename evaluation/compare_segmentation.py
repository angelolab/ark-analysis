import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# compare segmentation from different models
base_dir ='/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/'
save_dir = ''
file_paths = [base_dir + '20190917_naming/figs/dataframe.pkl', base_dir + '20190917_naming/figs/dataframe.pkl']
names = ["example1", "example2"]

predicted_errors = ["split", "merged", "low_quality"]
contour_errors = ["missing"]


# set up xr to hold accuracy metrics
vals = np.zeros((len(file_paths), len(predicted_errors) + len(contour_errors)))
error_xr = xr.DataArray(vals, coords=[names, predicted_errors + contour_errors], dims=["algos", "errors"])

# loop through each pandas array
for i in range(len(file_paths)):

    data = pd.read_pickle(file_paths[i])

    # loop through each error type
    for j in range(error_xr.shape[1]):
        # figure out whether the error is based on counting predicted or contour cell errors
        current_error = error_xr.coords["errors"].values[j]
        if current_error in predicted_errors:
            target_col = "predicted_cell"
        else:
            target_col = "contour_cell"

        error_xr.values[i, j] = len(set(data.loc[data[current_error], target_col]))
    error_xr.values[i, :] = error_xr.values[i, :] / len(np.unique(data["predicted_cell"]))


# create bar plots for each error type
for i in range(error_xr.shape[1]):
    current_error = error_xr.coords["errors"].values[i]
    position = range(error_xr.shape[0])
    fig, ax = plt.subplots(1, 1)
    ax.bar(position, error_xr.loc[:, current_error].values)

    ax.set_xticks(position)
    ax.set_xticklabels(error_xr["algos"].values)
    ax.set_title("Fraction of cells {}".format(current_error))
    fig.savefig(save_dir + current_error + ".tiff")


