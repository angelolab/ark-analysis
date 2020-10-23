## Data type information

The `ark` repo has different data structures for different data types. Below is a brief glossary highlighting the most important ones. 

---

Name: distance matrix  
Type: numpy.array  
Created by: [calc_dist_matrix](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.utils.html#ark.utils.spatial_analysis_utils.calc_dist_matrix)   
Used by: [calculate_channel_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment), [calculate_cluster_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment)  
Shape: 2D matrix of num_cells x num_cells  

Description: Many of the spatial analysis functions in the `analysis` module use distances between cells to compute interesting spatial properties. The distance matrix holds that information. Each matrix is a square array, where the value of cell (**i**, **j**) in the matrix represents the distance between cell **i** and cell **j**.  

Note: `calc_dist_matrix` produces a dictionary of distancs matrixes; each distance matrix takes the form described above

---

Name: segmentation labels


---

Name: 
