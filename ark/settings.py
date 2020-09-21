# hope u liek capital letters

# segmented csv column names
CELL_SIZE = 'cell_size'

PRE_CHANNEL_COLS = [
    CELL_SIZE,
]

CELL_LABEL = 'label'
AREA = 'area'
ECCENTRICITY = 'eccentricity'
MAJ_AXIS_LENGTH = 'major_axis_length'
MIN_AXIS_LENGTH = 'minor_axis_length'
PERIMITER = 'perimeter'
FOV_ID = 'SampleID'
CELL_TYPE = 'cell_type'
CLUSTER_ID = 'FlowSOM_ID'
PATIENT_ID = 'PatientID'

POST_CHANNEL_COLS = [
    CELL_LABEL,
    AREA,
    ECCENTRICITY,
    MAJ_AXIS_LENGTH,
    MIN_AXIS_LENGTH,
    PERIMITER,
    FOV_ID,
]

CLUSTERED_POST_CHANNEL_COLS = POST_CHANNEL_COLS + [
    CLUSTER_ID,
    CELL_TYPE,
]
