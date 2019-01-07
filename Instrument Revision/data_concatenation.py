import os
import pandas as pd
import numpy as np


# import data generated from segmenting cytoplasm and nucleus separately. Combine them into a single file for FCS plotting


path = '/Users/noahgreenwald/Documents/MIBI_Data/revisions/Data/181222_HighResDFCI/no_aggregates/'

csv_files = os.listdir(path + "fcs_nuc")
csv_files = [x for x in csv_files if '.csv' in x]
csv_files = [x for x in csv_files if 'Scaled_Transformed' not in x]

headers = pd.read_csv(path + "/fcs_nuc/col_names.txt", sep="\t")
header_names = headers.columns.values[0:-1]

for file in csv_files:
    nuc_data = pd.read_csv(path + "fcs_nuc/" + file, header=None)
    nuc_data.columns = header_names + "_nuc"

    cyto_data = pd.read_csv(path + "fcs_cyto/" + file, header = None)
    cyto_data.columns = header_names + "_cyto"

    combined_data = pd.concat((nuc_data, cyto_data), axis=1)
    combined_data.to_csv(path + "fcs_combined/" + file)