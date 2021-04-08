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

Description: This holds the extracted single cell data. Each row is a unique cell, and each column is a feature for that cell. This includes the counts for each marker, morphological information, and information to link that cell back to the original image such as segmentation id and FOV.  

For each cell, these are the specific morphology metrics computed:

* `cell_size`: the signal intensity. May not be the same as area depending on the signal extraction method used.
* `area`: number of pixels of the region
* `eccentricity`: eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
* `major_axis_length`: the length of the major axis of the ellipse that has the same normalized second central moments as the region
* `minor_axis_length`: the length of the minor axis of the ellipse that has the same normalized second central moments as the region.
* `perimeter`: perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity
* `convex_area`: the area of the convex hull
* `equivalent_diameter`: the diameter of the circle with the same area as the cell
* `centroid-0`: the x-coordinate of the centroid
* `centroid-1`: the y-coordinate of the centroid
* `major_minor_axis_ratio`: the major axis length divided by the minor axis length
* `perim_square_over_area`: the square of the perimeter divided by the area
* `major_axis_equiv_diam_ratio`: the major axis length divided by the equivalent diameter
* `convex_hull_resid`: the difference between the convex area and the area divided by the convex area
* `centroid_dif`: the normalized euclidian distance between the cell centroid and the corresponding convex hull centroid
* `num_concavities`
* `nc_ratio`: for nuclear segmentation only. The nuclear area divided by the total area.

---

Name: distance matrix  
Type: numpy.array  
Created by: [calc_dist_matrix](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.utils.html#ark.utils.spatial_analysis_utils.calc_dist_matrix)   
Used by: [calculate_channel_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment), [calculate_cluster_spatial_enrichment](https://ark-analysis.readthedocs.io/en/latest/_markdown/ark.analysis.html#ark.analysis.spatial_analysis.calculate_channel_spatial_enrichment)  
Shape: 2D matrix of num_cells x num_cells  

Description: Many of the spatial analysis functions in the `analysis` module use distances between cells to compute interesting spatial properties. The distance matrix holds that information. Each matrix is a square array, where the value of cell (**i**, **j**) in the matrix represents the distance between cell **i** and cell **j**.  

Note: `calc_dist_matrix` produces a dictionary of distancs matrixes; each distance matrix takes the form described above
