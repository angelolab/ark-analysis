# hope u liek capital letters

# default segmented csv column names
CELL_SIZE = 'cell_size'                     # cell size
CELL_LABEL = 'label'                        # cell label number (regionprops)
AREA = 'area'                               # cell area (regionprops)
ECCENTRICITY = 'eccentricity'               # cell eccentricity (regionprops)
MAJ_AXIS_LENGTH = 'major_axis_length'       # cell major axis length (regionprops)
MIN_AXIS_LENGTH = 'minor_axis_length'       # cell minor axis length (regionprops)
PERIMITER = 'perimeter'                     # cell perimiter (regionprops)
FOV_ID = 'SampleID'                         # cell's fov name
CELL_TYPE = 'cell_type'                     # cell type name (flowsom)
CLUSTER_ID = 'FlowSOM_ID'                   # cell cluster id (flowsom)
PATIENT_ID = 'PatientID'                    # cell's patient id

# standardized columns surrounding channel data
PRE_CHANNEL_COL = CELL_SIZE                 # last column before channel data
POST_CHANNEL_COL = CELL_LABEL               # first column after channel data
