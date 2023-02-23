import numpy as np

import synthetic_spatial_datagen


def test_generate_test_dist_matrix():
    # use default parameters
    sample_dist_mat = synthetic_spatial_datagen.generate_test_dist_matrix()

    # assert matrix symmetry
    assert np.allclose(sample_dist_mat.loc[np.arange(1, 301), np.arange(1, 301)].values,
                       sample_dist_mat.T.loc[np.arange(1, 301), np.arange(1, 301)].values,
                       rtol=1e-05, atol=1e-08)

    # assert the distributions are correct (AB < AC)
    assert sample_dist_mat.loc[np.arange(1, 101), np.arange(101, 201)].values.mean() < \
        sample_dist_mat.loc[np.arange(1, 101), np.arange(201, 301)].values.mean()


def test_generate_random_centroids():
    # generate some sample params
    size_img = (1024, 1024)

    num_A = 100
    num_B = 100
    num_C = 100

    mean_A_factor = 0.5
    mean_B_factor = 0.7
    mean_C_factor = 0.3

    cov_A = [[100, 0], [0, 100]]
    cov_B = [[100, 0], [0, 100]]
    cov_c = [[100, 0], [0, 100]]

    centroid_list = synthetic_spatial_datagen.generate_random_centroids(
        size_img=size_img,
        num_A=num_A,
        num_B=num_B,
        num_C=num_C,
        mean_A_factor=mean_A_factor,
        cov_A=cov_A,
        mean_B_factor=mean_B_factor,
        cov_B=cov_B,
        mean_C_factor=mean_C_factor,
        cov_C=cov_c
    )

    # try to extract non-duplicate centroids in the list
    _, centroid_counts = np.unique(centroid_list, axis=0, return_counts=True)
    non_dup_centroids = centroid_list[centroid_counts > 1]

    # assert that there are no duplicates in the list
    assert len(non_dup_centroids) == 0

    # separate x and y coords
    x_coords = centroid_list[:, 0]
    y_coords = centroid_list[:, 1]

    # assert the x and y coordinates are in range
    assert len(x_coords[(x_coords < 0) & (x_coords >= size_img[0])]) == 0
    assert len(y_coords[(y_coords < 0) & (y_coords >= size_img[0])]) == 0


def test_generate_test_label_map():
    # generate test data
    sample_img_xr = synthetic_spatial_datagen.generate_test_label_map()

    # flatten and remove all non-centroids for testing purposes
    label_map = np.stack(sample_img_xr[0, :, :, 0])
    label_map_flat = label_map.flatten()
    label_map_flat = label_map_flat[label_map_flat > 0]

    # all centroids must have a unique id
    _, label_map_id_counts = np.unique(label_map_flat, return_counts=True)
    assert len(label_map_flat[label_map_id_counts > 1]) == 0


def test_generate_two_cell_seg_mask():
    cell_radius = 10

    sample_segmentation_mask, sample_cell_centers = \
        synthetic_spatial_datagen.generate_two_cell_seg_mask(cell_radius=cell_radius)

    # assert that our labels are just 0, 1, and 2
    assert set(sample_segmentation_mask.flatten().tolist()) == set([0, 1, 2])

    # assert that centers are labeled correctly
    assert sample_segmentation_mask[sample_cell_centers[1][0], sample_cell_centers[1][0]] == 1
    assert sample_segmentation_mask[sample_cell_centers[2][0], sample_cell_centers[2][1]] == 2

    # assert that the cells are next to each other we only
    assert sample_segmentation_mask[sample_cell_centers[1][0], sample_cell_centers[1][1] + 10] == 2
    assert sample_segmentation_mask[sample_cell_centers[2][0], sample_cell_centers[2][1] - 10] == 1


# TODO: after jitter is added, test status of signal at nucleus border
def test_generate_two_cell_test_nuclear_signal():
    cell_radius = 10
    nuc_radius = 3
    nuc_signal_strength = 10
    nuc_uncertainty_length = 0

    sample_segmentation_mask, sample_cell_centers = \
        synthetic_spatial_datagen.generate_two_cell_seg_mask(cell_radius=cell_radius)

    sample_nuclear_signal = \
        synthetic_spatial_datagen.generate_two_cell_nuc_signal(
            segmentation_mask=sample_segmentation_mask,
            cell_centers=sample_cell_centers,
            nuc_radius=nuc_radius,
            nuc_signal_strength=nuc_signal_strength,
            nuc_uncertainty_length=nuc_uncertainty_length
        )

    # assert that our nucleus center is labeled properly
    assert sample_nuclear_signal[sample_cell_centers[1][0], sample_cell_centers[1][1]] == 10

    # add memb_uncertainty
    nuc_uncertainty_length = 1
    sample_nuclear_signal = \
        synthetic_spatial_datagen.generate_two_cell_nuc_signal(
            segmentation_mask=sample_segmentation_mask,
            cell_centers=sample_cell_centers,
            nuc_radius=nuc_radius,
            nuc_signal_strength=nuc_signal_strength,
            nuc_uncertainty_length=nuc_uncertainty_length
        )

    assert sample_nuclear_signal[sample_cell_centers[1][0], sample_cell_centers[1][1]] == 10


# TODO: after jitter is added, test status of signal at membrane border
def test_generate_two_cell_memb_signal():
    cell_radius = 10
    memb_thickness = 5
    memb_signal_strength = 10
    memb_uncertainty_length = 0

    sample_segmentation_mask, sample_cell_centers = \
        synthetic_spatial_datagen.generate_two_cell_seg_mask(cell_radius=cell_radius)

    sample_membrane_signal = \
        synthetic_spatial_datagen.generate_two_cell_memb_signal(
            segmentation_mask=sample_segmentation_mask,
            cell_centers=sample_cell_centers,
            cell_radius=cell_radius,
            memb_thickness=memb_thickness,
            memb_signal_strength=memb_signal_strength,
            memb_uncertainty_length=memb_uncertainty_length
        )

    # assert that our membrane inner edge is being labeled correctly
    assert sample_membrane_signal[
        sample_cell_centers[2][0],
        sample_cell_centers[2][1] - memb_thickness
    ] == 10

    # include a test with memb_uncertainty
    memb_uncertainty_length = 1
    sample_membrane_signal = \
        synthetic_spatial_datagen.generate_two_cell_memb_signal(
            segmentation_mask=sample_segmentation_mask,
            cell_centers=sample_cell_centers,
            cell_radius=cell_radius,
            memb_thickness=memb_thickness,
            memb_signal_strength=memb_signal_strength,
            memb_uncertainty_length=memb_uncertainty_length
        )

    assert sample_membrane_signal[
        sample_cell_centers[2][0],
        sample_cell_centers[2][1] - (memb_thickness + memb_uncertainty_length)
    ] == 10


def test_generate_two_cell_chan_data():
    _, sample_channel_data = \
        synthetic_spatial_datagen.generate_two_cell_chan_data()

    # must have both nuclear and membrane channels
    assert sample_channel_data.shape[2] == 2

    # assert that correct nuclear and membrane signal labeled
    assert set(sample_channel_data[:, :, 0].flatten().tolist()) == set([0, 10])
    assert set(sample_channel_data[:, :, 1].flatten().tolist()) == set([0, 10])

    # include a test with different nuclear and membrane signal strengths
    _, sample_channel_data = \
        synthetic_spatial_datagen.generate_two_cell_chan_data(
            nuc_signal_strength=10,
            memb_signal_strength=100
        )

    assert set(sample_channel_data[:, :, 0].flatten().tolist()) == set([0, 10])
    assert set(sample_channel_data[:, :, 1].flatten().tolist()) == set([0, 100])
