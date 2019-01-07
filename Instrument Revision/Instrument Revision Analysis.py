import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats.stats as stats
import scipy
import copy
import math
import numpy as np
from PIL import Image

import statistics

directory = ["/Users/noahgreenwald/Documents/MIBI_Data/revisions/Data/RemeasureTNBC/180502_TNBCrescansForInstrumentPaper/Point1/TIFsNoNoise",
             "/Users/noahgreenwald/Documents/MIBI_Data/revisions/Data/RemeasureTNBC/180502_TNBCrescansForInstrumentPaper/Point2/TIFsNoNoise",
              "/Users/noahgreenwald/Documents/MIBI_Data/revisions/Data/RemeasureTNBC/180502_TNBCrescansForInstrumentPaper/Point3/TIFsNoNoise"]

directory_old = ["/Users/noahgreenwald/Documents/MIBI_Data/revisions/Data/RemeasureTNBC/P1-old", "/Users/noahgreenwald/Documents/MIBI_Data/revisions/Data/RemeasureTNBC/P2-old",
                 "/Users/noahgreenwald/Documents/MIBI_Data/revisions/Data/RemeasureTNBC/P4-old"]

os.chdir("/Users/noahgreenwald/Documents/Grad School/Lab/InstrumentRevision/")
crop_dim = [[746,1770, 316,1340], [806,1830, 316,1340], [500,1524, 500,1524]]

# code for figuring out where to crop image
# dna_old = plt.imread(directory_old[0] + "/SMA.tif")
# dna_new = plt.imread(directory[0] + "/SMA.tif")
# plt.figure()
# plt.imshow(dna_new)
# plt.figure()
# crop = crop_dim[0]
# plt.imshow(dna_old[crop[0]:crop[1], crop[2]:crop[3]])


# import data for analysis
channels = os.listdir(directory[2])

# naturally abundant elements
outliers = ["Fe.tif", "C.tif", "Ca.tif", "Si.tif", "Ta.tif", "Au.tif", "Background.tif", "P.tif", "Na.tif"]

# values[np.logical_and(values["old_value"] < 400, values["new_value"] < 400)]
# channels without signal
outliers = outliers + ["B7H3.tif", "CD163.tif", "Keratin17.tif", "CSF-1R.tif", "OX40.tif"]

# p53 aggregates only
outliers = outliers + ["p53.tif"]

channels = [x for x in channels if x not in outliers]

# create cropped version of tifs
for pos, point in enumerate(directory_old):
    for image in os.listdir(directory[0]):
        if '.tif' in image:
            img = plt.imread(point + "/" + image)
            crop = crop_dim[pos]
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
            name = image.strip('.tif') + '_cropped.tif'
            raw_vals = Image.fromarray(img)
            raw_vals.save(point + "/" + name)


# old code for comparing intensity across the entire image

# values = pd.DataFrame(columns=["old_value", "new_value", "ratio", "point"], dtype="float")
# for point in range(len(directory_old)):
#     for chan in channels:
#         mtrx_1 = plt.imread(directory[point] + "/" + chan)
#         mtrx_2 = plt.imread(directory_old[point] + "/" + chan)
#         crop = crop_dim[point]
#         mtrx_2 = mtrx_2[crop[0]:crop[1], crop[2]:crop[3]]
#         values = values.append({"old_value":(mtrx_2.sum() + 1), "new_value":(mtrx_1.sum() + 1)}, ignore_index=True)
#
# values["point"] = ["1"] * len(channels) + ["2"] * len(channels) + ["3"] * len(channels)
#
# values.index = channels * 3
# values["ratio"] = values["new_value"] / values["old_value"]
# values["new_log"] = [math.log10(x) for x in values["new_value"]]
# values["old_log"] = [math.log10(x) for x in values["old_value"]]
#
# values_scrambled = copy.deepcopy(values)
#
# # move values from same channel down one point
# values_scrambled["old_value"].iloc[0:int(2*(len(values)/3))] = values["old_value"].iloc[int(len(values) / 3): len(values)].values
# values_scrambled["old_value"].iloc[int(2*(len(values)/3)):len(values)] = values["old_value"].iloc[0:int(len(values) / 3)].values
#
#
# values_scrambled["old_log"].iloc[0:int(2*(len(values)/3))] = values["old_log"].iloc[int(len(values) / 3): len(values)]
# values_scrambled["old_log"].iloc[int(2*(len(values)/3)):len(values)] = values["old_log"].iloc[0:int(len(values) / 3)]


# plt.scatter(values_scrambled["old_log"], values_scrambled["new_log"])
# plt.scatter(values["old_log"], values["new_log"])
#
#
# save_path = "/Users/noahgreenwald/Documents/Grad School/Lab/InstrumentRevision/"
# f = plt.figure()
# plt.scatter(values["old_log"], values["new_log"])
# plt.title("Logged intensity pre vs post")
# plt.xlabel("old value")
# plt.ylabel("new value")
# f.savefig(save_path + "pre_vs_post_log.pdf")
#
# f = plt.figure()
# plt.scatter(values_scrambled["old_log"], values_scrambled["new_log"])
# plt.title("Logged intensity pre vs post scramble")
# plt.xlabel("old value")
# plt.ylabel("new value")
# f.savefig(save_path + "pre_vs_post_scrambled_log.pdf")


# stats.pearsonr(values["old_log"], values["new_log"])
# stats.pearsonr(values_scrambled["old_log"], values_scrambled["new_log"])


# # break each image up into mutliple pieces, do correlation across peices
# values_grid_16 = pd.DataFrame(columns=["old_value", "new_value", "ratio", "point", "position"], dtype="float")
# values_grid_8 = pd.DataFrame(columns=["old_value", "new_value", "ratio", "point", "position"], dtype="float")
# values_grid_4 = pd.DataFrame(columns=["old_value", "new_value", "ratio", "point", "position"], dtype="float")

values_grid = pd.DataFrame(columns=["old_value", "new_value", "ratio", "point", "position"], dtype="float")

for point in range(len(directory_old)):
    for chan in channels:
        mtrx_1 = plt.imread(directory[point] + "/" + chan)
        mtrx_2 = plt.imread(directory_old[point] + "/" + chan)
        crop = crop_dim[point]
        mtrx_2 = mtrx_2[crop[0]:crop[1], crop[2]:crop[3]]

        split = 10
        step_size = int(1024 / split)

        for row in range(split):
            for col in range(split):
                sub_mtrx_1 = mtrx_1[(row * step_size):((row + 1) * step_size), (col * step_size):((col + 1) * step_size)]
                sub_mtrx_2 = mtrx_2[(row * step_size):((row + 1) * step_size), (col * step_size):((col + 1) * step_size)]

                values_grid = values_grid.append({"old_value": (sub_mtrx_2.sum() + 1),
                                                  "new_value": (sub_mtrx_1.sum() + 1),
                                                  "position": (str(row) + str(col)),
                                                  "point": point}, ignore_index=True)

# set label to be correct channel
# values_grid_4.index = [x for x in channels for _ in range(16)] * 3
# values_grid_8.index = [x for x in channels for _ in range(64)] * 3
# values_grid_16.index = [x for x in channels for _ in range(256)] * 3

values_grid.index = [x for x in channels for _ in range(100)] * 3

#
# values_grid["new_log"] = [math.log10(x) for x in values_grid["new_value"]]
# values_grid["old_log"] = [math.log10(x) for x in values_grid["old_value"]]
#
#
# values_grid_scrambled = copy.deepcopy(values_grid)
#
# # move values from same channel down one point
# values_grid_scrambled["old_value"].iloc[0:int(2*(len(values_grid)/3))] = values_grid["old_value"].iloc[int(len(values_grid) / 3): len(values_grid)].values
#
#
# values_grid_scrambled["old_value"].iloc[int(2*(len(values_grid)/3)):len(values_grid)] = values_grid["old_value"].iloc[0:int(len(values_grid) / 3)].values
#
#
# values_grid_scrambled["old_log"].iloc[0:int(2*(len(values_grid)/3))] = values_grid["old_log"].iloc[int(len(values_grid) / 3): len(values_grid)].values
# values_grid_scrambled["old_log"].iloc[int(2*(len(values_grid)/3)):len(values_grid)] = values_grid["old_log"].iloc[0:int(len(values_grid) / 3)].values
#
#
# plt.scatter(values_grid_scrambled["old_log"], values_grid_scrambled["new_log"])
#
# stats.pearsonr(values_grid["old_log"], values_grid["new_log"])
# stats.pearsonr(values_grid_scrambled["old_log"], values_grid_scrambled["new_log"])

# generate channel specific r values
# grids = ["4x4", "8x8", "16x16"]
#
# r_values = pd.DataFrame(columns=["channel", "point", "intensity_4", "correlation_4", "intensity_8", "correlation_8",
#                                  "intensity_16", "correlation_16"], dtype="float")
#
# for point in range(len(directory_old)):
#     for chan in channels:
#         old_val = [values_grid_4.loc[np.logical_and(values_grid_4.index == chan, values_grid_4["point"] == point), "old_value"],
#                    values_grid_8.loc[np.logical_and(values_grid_8.index == chan, values_grid_8["point"] == point), "old_value"],
#                    values_grid_16.loc[np.logical_and(values_grid_16.index == chan, values_grid_16["point"] == point), "old_value"]]
#
#         new_val = [values_grid_4.loc[np.logical_and(values_grid_4.index == chan, values_grid_4["point"] == point), "new_value"],
#                    values_grid_8.loc[np.logical_and(values_grid_8.index == chan, values_grid_8["point"] == point), "new_value"],
#                    values_grid_16.loc[np.logical_and(values_grid_16.index == chan, values_grid_16["point"] == point), "new_value"]]
#
#         # for size in range(len(grids)):
#         #     f = plt.figure()
#         #     plt.scatter(old_val[size], new_val[size])
#         #     plt.title("Intensity pre vs post " + grids[size] + " for point" + str(point) + " and channel " + chan[:-4])
#         #     plt.xlabel("old value")
#         #     plt.ylabel("new value")
#         #     f.savefig(save_path + "pre_vs_post_" + chan[:-4] + "_point_" + str(point) + "_" + grids[size] + ".pdf")
#         #     plt.close()
#
#         r_val = [stats.pearsonr(old_val[0], new_val[0])[0], stats.pearsonr(old_val[1], new_val[1])[0],
#                  stats.pearsonr(old_val[2], new_val[2])[0]]
#
#         # change correlation value back to zero if nan
#         r_val = [0 if np.isnan(x) else x for x in r_val]
#         r_values = r_values.append({"channel": chan,
#                                     "point": point,
#                                     "intensity_4": sum(old_val[0].values + new_val[0].values),
#                                     "correlation_4": r_val[0],
#                                     "intensity_8": sum(old_val[1].values + new_val[1].values),
#                                     "correlation_8": r_val[1],
#                                     "intensity_16": sum(old_val[2].values + new_val[2].values),
#                                     "correlation_16": r_val[2]}, ignore_index=True)
#

# new version that picks single value

r_values = pd.DataFrame(columns=["channel", "point", "intensity", "correlation", "variance"], dtype="float")

for point in range(len(directory_old)):
    for chan in channels:

        #old_val = values_grid.loc[np.logical_and(values_grid.index == chan, values_grid["point"] == point), "old_value"]
        #new_val = values_grid.loc[np.logical_and(values_grid.index == chan, values_grid["point"] == point), "new_value"]

        old_val = values_grid.loc[(values_grid.index == chan), "old_value"]
        new_val = values_grid.loc[(values_grid.index == chan), "new_value"]

        r_val = stats.pearsonr(old_val, new_val)[0]

        # change correlation value back to zero if nan
        if np.isnan(r_val):
            r_val = 0

        coef = scipy.stats.variation(old_val)

        r_values = r_values.append({"channel": chan,
                                    "point": point,
                                    "intensity": sum(old_val.values + new_val.values),
                                    "correlation": r_val,
                                    "variance": coef}, ignore_index=True)


# plot correlation at each point against total intensity
r_values_plot = copy.deepcopy(r_values)
r_values_plot = r_values_plot[r_values_plot["variance"] > 0.5]
r_values_plot = r_values_plot[r_values_plot["intensity"] > 5000]

f = plt.figure()
plt.scatter(r_values_plot["intensity"], r_values_plot["correlation"])
plt.xlabel("total signal intensity")
plt.ylabel("correlation")
plt.xscale("log")
f.savefig("correlation_vs_intensity_10x10_merged.pdf")
plt.close()


r_values_plot.to_csv("r2_values.csv")

stats.pearsonr([math.log10(x) for x in r_values_plot["intensity"]], r_values_plot["correlation"])


# highlight scatter pre vs post for specific channels

old_scatter = values_grid.loc[np.logical_and(values_grid.index == "CD45.tif", values_grid["point"] == 0.0), "old_value"]
new_scatter = values_grid.loc[np.logical_and(values_grid.index == "CD45.tif", values_grid["point"] == 0.0), "new_value"]

old_scatter = values_grid.loc[(values_grid.index == "CD45.tif"), "old_value"]
new_scatter = values_grid.loc[(values_grid.index == "CD45.tif"), "new_value"]

f = plt.figure()
plt.scatter(old_scatter, new_scatter)
plt.xlabel("Original Scan")
plt.ylabel("Rescan")
f.savefig("CD45_Merged_Example.pdf")
plt.close()
stats.pearsonr(old_scatter, new_scatter)

fig, ax = plt.subplots()
ax.scatter(r_values_plot["intensity"], r_values_plot["correlation"])
plt.xscale("log")

for i, txt in enumerate(r_values_plot["channel"]):
    ax.annotate(txt + " point " + str(r_values_plot["point"].values[i]), (r_values_plot["intensity"].values[i], r_values_plot["correlation"].values[i]), size = 10)
fig.set_size_inches(20,20)
fig.savefig("correlation_vs_intensity_10x10_labeled.pdf")
plt.close()




# plot old and new tifs for each point for all channels to examine poor correlatoins

for x in channels:

    chan_comp = "/" + x
    temp_old = [plt.imread(directory_old[0] + chan_comp), plt.imread(directory_old[1] + chan_comp),
                plt.imread(directory_old[2] + chan_comp)]

    temp_new = [plt.imread(directory[0] + chan_comp), plt.imread(directory[1] + chan_comp),
                plt.imread(directory[2] + chan_comp)]

    crop0 = crop_dim[0]
    crop1 = crop_dim[1]
    crop2 = crop_dim[2]

    fig, axs = plt.subplots(2, 3, figsize=(15, 15), sharey=True)
    axs[0, 0].imshow(temp_old[0][crop0[0]:crop0[1], crop0[2]:crop0[3]])
    axs[0, 0].set_title("point 0 old")

    axs[0, 1].imshow(temp_old[1][crop1[0]:crop1[1], crop1[2]:crop1[3]])
    axs[0, 1].set_title("point 1 old")

    axs[0, 2].imshow(temp_old[2][crop2[0]:crop2[1], crop2[2]:crop2[3]])
    axs[0, 2].set_title("point 2 old")


    axs[1, 0].imshow(temp_new[0])
    axs[1, 0].set_title("point 0 new")

    axs[1, 1].imshow(temp_new[1])
    axs[1, 1].set_title("point 1 new")

    axs[1, 2].imshow(temp_new[2])
    axs[1, 2].set_title("point 2 new")

    fig.suptitle(str(x))

    fig.savefig(save_path + chan_comp)
    plt.close()


