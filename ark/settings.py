# hope u liek capital letters

# default cell table column names
CELL_SIZE = 'cell_size'  # cell size (number of pixels in the cell)
CELL_LABEL = 'label'  # cell label number (regionprops)
FOV_ID = 'SampleID'  # cell's fov name
CELL_TYPE = 'cell_type'  # cell type name (flowsom)
CLUSTER_ID = 'FlowSOM_ID'  # cell cluster id (flowsom)
PATIENT_ID = 'PatientID'  # cell's patient id
KMEANS_CLUSTER = 'cluster_labels'  # generated cluster column name
CENTROID_0 = 'centroid-0'  # cell centroid x-coordinate
CENTROID_1 = 'centroid-1'  # cell centroid y-coordinate

# standardized columns surrounding channel data
PRE_CHANNEL_COL = CELL_SIZE  # last column before channel data
POST_CHANNEL_COL = CELL_LABEL  # first column after channel data

# regionprops extraction
REGIONPROPS_BASE = ['label', 'area', 'eccentricity', 'major_axis_length',
                    'minor_axis_length', 'perimeter', 'centroid', 'convex_area',
                    'equivalent_diameter']
REGIONPROPS_SINGLE_COMP = ['major_minor_axis_ratio', 'perim_square_over_area',
                           'major_axis_equiv_diam_ratio', 'convex_hull_resid',
                           'centroid_dif', 'num_concavities']
REGIONPROPS_MULTI_COMP = ['nc_ratio']

# spatial-LDA minimum required columns
BASE_COLS = [FOV_ID, CELL_LABEL, CELL_SIZE, CENTROID_0, CENTROID_1, CLUSTER_ID, KMEANS_CLUSTER]

# spatial_lda topic EDA key names
EDA_KEYS = ['inertia', 'silhouette', 'gap_stat', 'gap_sds', 'percent_var_exp', 'cell_counts']


