import os
import pandas as pd

# import data generated from matlab segmentation pipeline and combine each individual into single master file

path = '/Users/noahgreenwald/Documents/MIBI_Data/selena/20191215_GBM_Cohort/TA_551_test/single_cell_output/'

csv_files = os.listdir(path)
csv_files = [x for x in csv_files if '.csv' in x]
csv_files = [x for x in csv_files if 'transformed' not in x and 'Col' not in x]

for file in csv_files:
    print(file)

    if file == csv_files[0]:
        # first one, create master array
        temp_data = pd.read_csv(path + file, header=0, sep=",")
        temp_data["point"] = file
        combined_data = temp_data
    else:
        temp_data = pd.read_csv(path + file, header=0, sep=",")
        temp_data["point"] = file
        combined_data = pd.concat((combined_data, temp_data), axis=0, ignore_index=True)

combined_data.to_csv(path + "_combined_data.csv")