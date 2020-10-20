# hope u liek capital letters

# default segmented csv column names
CELL_SIZE = 'cell_size'                     # cell size
CELL_LABEL = 'label'                        # cell label number (regionprops)
FOV_ID = 'SampleID'                         # cell's fov name
CELL_TYPE = 'cell_type'                     # cell type name (flowsom)
CLUSTER_ID = 'FlowSOM_ID'                   # cell cluster id (flowsom)
PATIENT_ID = 'PatientID'                    # cell's patient id
KMEANS_CLUSTER = 'cluster_labels'           # generated cluster column name

# standardized columns surrounding channel data
PRE_CHANNEL_COL = CELL_SIZE                 # last column before channel data
POST_CHANNEL_COL = CELL_LABEL               # first column after channel data
