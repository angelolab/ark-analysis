## Data type information

The `ark` repo has different data structures for different data types. Below is a brief glossary highlighting the most important ones. 

---

Name: segmentation labels  
Type: xarray.DataArray  
Created by: [create_deepcell_output](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.utils.html#ark.utils.deepcell_service_utils.create_deepcell_output)  
Used by: [calc_dist_matrix](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.utils.html#ark.utils.spatial_analysis_utils.calc_dist_matrix)  
Shape: 4D matrix of fov x num_cells x num_cells x compartment  

Description: This is how segmentation predictions are represented. Each cell is assinged a unique integer value. All of the pixels belonging to that cell have the same value. e.g. all pixels belonging to cell 1 have a value of 1, all pixels belonging to cell 5 have a value of 5, etc. This data structure holds the unique set of labels for each FOV on the first axis. IF there are multiple labels per FOV, for example nuclear and whole-cell, these are stored on the last axis

---

Name: cell table  
Type: pandas.DataFrame  
Created by: [generate_cell_table](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.segmentation.html#ark.segmentation.marker_quantification.generate_cell_table)   
Used by: [calculate_channel_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment), [calculate_cluster_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment),
[create_neighborhood_matrix](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.segmentation.html#ark.segmentation.marker_quantification.generate_cell_data)  
Shape: 2D matrix of cells x cell_features  

Description: This holds the extracted single cell data. Each row is a unique cell, and each column is a feature for that cell. This includes the counts for each marker, morphological information such as area and diameter, and information to link that cell back to the original image such as segmentation id and FOV.

---

Name: distance matrix  
Type: numpy.array  
Created by: [calc_dist_matrix](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.utils.html#ark.utils.spatial_analysis_utils.calc_dist_matrix)   
Used by: [calculate_channel_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment), [calculate_cluster_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment)  
Shape: 2D matrix of num_cells x num_cells  

Description: Many of the spatial analysis functions in the `analysis` module use distances between cells to compute interesting spatial properties. The distance matrix holds that information. Each matrix is a square array, where the value of cell (**i**, **j**) in the matrix represents the distance between cell **i** and cell **j**.  

Note: `calc_dist_matrix` produces a dictionary of distancs matrixes; each distance matrix takes the form described above
