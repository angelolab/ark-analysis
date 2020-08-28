import numpy as np

from ark.utils import synthetic_spatial_datagen


def test_generate_test_dist_matrix():
    # this function tests the functionality of a random initialization of a
    # distance matrix directly.
    # the tests are fairly basic for now but will be expanded on.
    # for now, we just want to ensure two things:
    #   matrix symmetry: a property of distance matrices
    #   mean AB > mean AC: the mean distance between cells of types A and B
    #                      is greater than that of types A and C

    # we'll be using the default parameters provided in the functions
    # except for the random seed, which we specify
    sample_dist_mat = synthetic_spatial_datagen.generate_test_dist_matrix()

    # assert matrix symmetry
    assert np.allclose(sample_dist_mat.loc[np.arange(1, 301), np.arange(1, 301)].values,
                       sample_dist_mat.T.loc[np.arange(1, 301), np.arange(1, 301)].values,
                       rtol=1e-05, atol=1e-08)

    # assert the average of the distance between A and B is smaller
    # than the average of the distance between A and C.
    # this may not eliminate the possibility that the null is proved true
    # but it's definitely a great check that can ensure greater success
    assert sample_dist_mat.loc[np.arange(1, 101), np.arange(101, 201)].values.mean() < \
        sample_dist_mat.loc[np.arange(1, 101), np.arange(201, 301)].values.mean()


def test_generate_random_centroids():
    # this function tests the functionality of a random initialization of centroids
    # the tests are fairly basic for now but will be expanded on
    # for the time being, all we do are the following:
    #   test that there are no duplicates in centroid_list
    #   test that all the centroids generated are in range

    # generate some sample stats to pass into generate_random_centroids to get our values
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
    # this function tests the general nature of the random label map generated
    # the crux of which are the centroids generated from test_generate_random_centroids
    # the tests are fairly basic for now but will be expanded on
    # for now, all we're doing is checking if we're labeling each centroid with a unique label
    # we also check and see that each label appears in the label_map

    # generate test data
    sample_img_xr = synthetic_spatial_datagen.generate_test_label_map()

    # all we're looking at is the label map
    # we flatten and remove all non-centroids for testing purposes
    label_map = np.stack(sample_img_xr[0, :, :, 0])
    label_map_flat = label_map.flatten()
    label_map_flat = label_map_flat[label_map_flat > 0]

    # need to assert that we're labeling all centroids with a unique id
    _, label_map_id_counts = np.unique(label_map_flat, return_counts=True)
    assert len(label_map_flat[label_map_id_counts > 1]) == 0


def test_generate_two_cell_test_segmentation_mask():
    # this function tests the functionality of generating the segmentation mask
    cell_radius = 10

    sample_segmentation_mask, sample_cell_centers = \
        synthetic_spatial_datagen.generate_two_cell_test_segmentation_mask(cell_radius=cell_radius)

    # assert that our labels are just blank, 1, and 2
    assert set(sample_segmentation_mask.flatten().tolist()) == set([0, 1, 2])

    # assert that our centers are being labeled correctly
    assert sample_segmentation_mask[sample_cell_centers[1][0], sample_cell_centers[1][0]] == 1
    assert sample_segmentation_mask[sample_cell_centers[2][0], sample_cell_centers[2][1]] == 2

    # using the default cell radius of 10, assert that the cells are next to each other we only
    # include the offset columnwise because that's how the cells are generated next to each other
    assert sample_segmentation_mask[sample_cell_centers[1][0], sample_cell_centers[1][1] + 10] == 2
    assert sample_segmentation_mask[sample_cell_centers[2][0], sample_cell_centers[2][1] - 10] == 1


def test_generate_two_cell_test_nuclear_signal():
    # this function tests the functionality of the nuclear-signal-generating portion of
    # the channel-level spatial analysis data
    cell_radius = 10
    nuc_radius = 3
    nuc_signal_strength = 10
    nuc_uncertainty_length = 0

    sample_segmentation_mask, sample_cell_centers = \
        synthetic_spatial_datagen.generate_two_cell_test_segmentation_mask(cell_radius=cell_radius)

    sample_nuclear_signal = \
        synthetic_spatial_datagen.generate_two_cell_test_nuclear_signal(
            segmentation_mask=sample_segmentation_mask,
            cell_centers=sample_cell_centers,
            nuc_radius=nuc_radius,
            nuc_signal_strength=nuc_signal_strength,
            nuc_uncertainty_length=nuc_uncertainty_length
        )

    # assert that our nucleus center is labeled properly
    # we only care about cell 1 because that is the only nuclear-level expression cell by default
    assert sample_nuclear_signal[sample_cell_centers[1][0], sample_cell_centers[1][1]] == 10

    # now include a test where we add memb_uncertainty
    nuc_uncertainty_length = 1
    sample_nuclear_signal = \
        synthetic_spatial_datagen.generate_two_cell_test_nuclear_signal(
            segmentation_mask=sample_segmentation_mask,
            cell_centers=sample_cell_centers,
            nuc_radius=nuc_radius,
            nuc_signal_strength=nuc_signal_strength,
            nuc_uncertainty_length=nuc_uncertainty_length
        )

    assert sample_nuclear_signal[sample_cell_centers[1][0], sample_cell_centers[1][1]] == 10

    # because we'll be jittering the signal eventually, we won't test the status of the signal at
    # the nucleus border this kind of hurts the nuc_uncertainty_length test but we'll revisit that
    # when the time comes


def test_generate_two_cell_test_membrane_signal():
    # this function tests the functionality of the membrane-signal-generating portion of
    # the channel-level spatial analysis data
    cell_radius = 10
    memb_thickness = 5
    memb_signal_strength = 10
    memb_uncertainty_length = 0

    sample_segmentation_mask, sample_cell_centers = \
        synthetic_spatial_datagen.generate_two_cell_test_segmentation_mask(cell_radius=cell_radius)

    sample_membrane_signal = \
        synthetic_spatial_datagen.generate_two_cell_test_membrane_signal(
            segmentation_mask=sample_segmentation_mask,
            cell_centers=sample_cell_centers,
            cell_radius=cell_radius,
            memb_thickness=memb_thickness,
            memb_signal_strength=memb_signal_strength,
            memb_uncertainty_length=memb_uncertainty_length
        )

    # assuming the default membrane diameter of 5, assert that our membrane inner edge is being
    # labeled correctly we only include the offset columnwise by choice: it could be done rowwise
    # as well we only care about cell 2 because that is the only membrane-level expression cell by
    # default
    assert sample_membrane_signal[
        sample_cell_centers[2][0],
        sample_cell_centers[2][1] - memb_thickness
    ] == 10

    # now include a test where we add memb_uncertainty
    memb_uncertainty_length = 1
    sample_membrane_signal = \
        synthetic_spatial_datagen.generate_two_cell_test_membrane_signal(
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

    # because we'll be jittering the signal eventually, we won't test the status of the signal at
    # the outer membrane border


def test_generate_two_cell_test_channel_synthetic_data():
    # this function tests the functionality of the overall signal-data creation process
    # with both the nuclear- and membrane-level channels
    # however, because we already do the sample_segmentation_mask test in a different function
    # we'll limit ourselves to testing the final channel data creation here

    _, sample_channel_data = \
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data()

    # assert that we've created both a nuclear and membrane channels
    assert sample_channel_data.shape[2] == 2

    # assert that we've only labeled nuclear and membrane signal with 0 or 10
    assert set(sample_channel_data[:, :, 0].flatten().tolist()) == set([0, 10])
    assert set(sample_channel_data[:, :, 1].flatten().tolist()) == set([0, 10])

    # now include a test where we set the nuclear and membrane signal strengths differently
    _, sample_channel_data = \
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data(
            nuc_signal_strength=10,
            memb_signal_strength=100
        )

    assert set(sample_channel_data[:, :, 0].flatten().tolist()) == set([0, 10])
    assert set(sample_channel_data[:, :, 1].flatten().tolist()) == set([0, 100])
