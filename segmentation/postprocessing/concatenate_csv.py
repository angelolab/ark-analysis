import os
import pandas as pd

# import data generated from matlab segmentation pipeline and combine each individual into single master file

path = '/Users/noahgreenwald/Documents/MIBI_Data/felix/20191001_cohort/fcs_3_pixel_3smooth/'

csv_files = os.listdir(path)
csv_files = [x for x in csv_files if '.txt' in x]
csv_files = [x for x in csv_files if 'Raw' not in x and 'Col' not in x]

headers = pd.read_csv(path + "Col_names.txt", sep="\t", header=None)
header_names = headers[0].values

for file in csv_files:

    if file == csv_files[0]:
        # first one, create master array
        temp_data = pd.read_csv(path + file, header=None, sep="\t")
        temp_data.columns = header_names
        temp_data["point"] = file
        combined_data = temp_data
    else:
        temp_data = pd.read_csv(path + file, header=None, sep="\t")
        temp_data.columns = header_names
        temp_data["point"] = file
        combined_data = pd.concat((combined_data, temp_data), axis=0, ignore_index=True)

combined_data.to_csv(path + "combined_csv.csv")