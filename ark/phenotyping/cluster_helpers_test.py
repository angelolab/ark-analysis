import feather
import numpy as np
import pandas as pd
import pathlib
import pytest
from typing import Iterator, List, Tuple

from ark.phenotyping.cluster_helpers import PixieConsensusCluster
from ark.utils.misc_utils import verify_same_elements


@pytest.fixture(scope="session")
def consensus_base_dir_gen(tmp_path_factory) -> Iterator[pathlib.Path]:
    """Creates the directory to hold all the test data needed for consensus clustering

    Args:
        tmp_path_factory (pytest.TempPathFactory):
            Temp dir to place consensus data in

    Yields:
        Iterator[pathlib.Path]:
            The path to place the consensus data in
    """

    consensus_output_dir = tmp_path_factory.mktemp("consensus_data")
    yield consensus_output_dir


@pytest.fixture(scope="session")
def pixel_input_gen(consensus_base_dir_gen) -> Iterator[Tuple[pathlib.Path, List[str]]]:
    """Generates sample pixel consensus data and save

    Args:
        consensus_base_dir_gen (pytest.fixture):
            The base dir to store the consensus data

    Yields:
        Iterator[Tuple[pathlib.Path, List[str]]]:
            Tuple containing path to the input to pixel SOM channel expression averages
            and expression columns
    """

    # define the paths, using a temporary directory as the motherbase
    output_file_path = consensus_base_dir_gen / "pixel_channel_avg_som_cluster.csv"

    # define the channel expression columns
    chan_cols = [f'chan{i}' for i in np.arange(1, 7)]

    # generate the sample data
    sample_pixel_consensus_data = pd.DataFrame(
        np.random.rand(100, 6), columns=chan_cols
    )

    # generate sample SOM cluster data values
    sample_pixel_consensus_data['pixel_som_cluster'] = np.arange(1, 101)

    # save the data
    sample_pixel_consensus_data.to_csv(output_file_path)

    yield output_file_path, chan_cols


@pytest.fixture(scope="session")
def cell_input_gen(consensus_base_dir_gen) -> Iterator[Tuple[pathlib.Path, List[str]]]:
    """Generates sample cell consensus data and save

    Args:
        consensus_base_dir_gen (pytest.fixture):
            The base dir to store the consensus data

    Yields:
        Iterator[Tuple[pathlib.Path, List[str]]]:
            Tuple containing path to the input to cell SOM count expression averages
            and expression columns
    """

    # define the paths, using a temporary directory as the motherbase
    output_file_path = consensus_base_dir_gen / "cell_som_cluster_avgs.csv"

    # define the pixel cluster count expression columns
    count_cols = [f'pixel_meta_cluster_{i}' for i in np.arange(1, 7)]

    # generate the sample data
    sample_cell_consensus_data = pd.DataFrame(
        np.random.rand(100, 6), columns=count_cols
    )

    # generate sample SOM cluster data values
    sample_cell_consensus_data['cell_som_cluster'] = np.arange(1, 101)

    # save the data
    sample_cell_consensus_data.to_csv(output_file_path)

    yield output_file_path, count_cols


@pytest.fixture(scope="session")
def pixel_cc_object(pixel_input_gen):
    yield PixieConsensusCluster(
        cluster_type='pixel', input_file=pixel_input_gen[0], columns=pixel_input_gen[1]
    )


@pytest.fixture(scope="session")
def cell_cc_object(cell_input_gen):
    yield PixieConsensusCluster(
        cluster_type='cell', input_file=cell_input_gen[0], columns=cell_input_gen[1]
    )


class TestPixieConsensusCluster:
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, pixel_cc_object, cell_cc_object):
        self.pixel_cc = pixel_cc_object
        self.cell_cc = cell_cc_object

    def test_bad_cluster_type(self):
        with pytest.raises(ValueError):
            PixieConsensusCluster(
                'bad_cluster_type', 'sample_input_file', ['col1', 'col2']
            )

    def test_pixel_scale_data(self):
        self.pixel_cc.scale_data()
        assert np.all(self.pixel_cc.input_data[self.pixel_cc.columns].values >= -3)
        assert np.all(self.pixel_cc.input_data[self.pixel_cc.columns].values <= 3)

    def test_cell_scale_data(self):
        self.cell_cc.scale_data()
        assert np.all(self.cell_cc.input_data[self.cell_cc.columns].values >= -3)
        assert np.all(self.cell_cc.input_data[self.cell_cc.columns].values <= 3)

    def test_run_pixel_consensus_clustering(self):
        self.pixel_cc.run_consensus_clustering()

        # assert we've created both internal Mk and bestK values for predict_data
        assert self.pixel_cc.cc.Mk is not None
        assert self.pixel_cc.cc.bestK is not None

    def test_run_cell_consensus_clustering(self):
        self.cell_cc.run_consensus_clustering()

        # assert we've created both internal Mk and bestK values for predict_data
        assert self.cell_cc.cc.Mk is not None
        assert self.cell_cc.cc.bestK is not None

    def test_generate_pixel_som_to_meta_map(self):
        self.pixel_cc.generate_som_to_meta_map()
        verify_same_elements(
            generated_meta_clusters=self.pixel_cc.mapping[self.pixel_cc.meta_col],
            required_meta_clusters=np.arange(1, 21)
        )

    def test_generate_cell_som_to_meta_map(self):
        self.cell_cc.generate_som_to_meta_map()
        verify_same_elements(
            generated_meta_clusters=self.cell_cc.mapping[self.cell_cc.meta_col],
            required_meta_clusters=np.arange(1, 21)
        )

    def test_save_pixel_som_to_meta_map(self):
        pixel_file = f'{self.pixel_cc.cluster_type}_clust_to_meta.feather'
        pixel_map_path = self.pixel_cc.input_file.parents[0] / pixel_file
        self.pixel_cc.save_som_to_meta_map(pixel_map_path)

        # assert we created the save path
        assert pixel_map_path.exists()

    def test_save_cell_som_to_meta_map(self):
        cell_file = f'{self.cell_cc.cluster_type}_clust_to_meta.feather'
        cell_map_path = self.cell_cc.input_file.parents[0] / cell_file
        self.cell_cc.save_som_to_meta_map(cell_map_path)

        # assert we created the save path
        assert cell_map_path.exists()

    def test_assign_pixel_consensus_labels(self):
        # generate sample external data with SOM labels
        sample_external_data = pd.DataFrame(
            np.random.rand(1000, 10)
        )
        sample_external_data[self.pixel_cc.som_col] = np.repeat(np.arange(1, 101), 10)

        labeled_external_data = self.pixel_cc.assign_consensus_labels(sample_external_data)

        # ensure we've created a meta cluster column
        assert self.pixel_cc.meta_col in sample_external_data.columns.values

        # ensure all the mappings match up
        external_mappings = labeled_external_data[
            [self.pixel_cc.som_col, self.pixel_cc.meta_col]
        ].copy()
        external_mappings = external_mappings.drop_duplicates().sort_values(
            by=self.pixel_cc.som_col
        )
        true_mappings = self.pixel_cc.mapping.sort_values(by=self.pixel_cc.som_col)
        assert np.all(external_mappings.values == true_mappings.values)

    def test_assign_cell_consensus_labels(self):
        # generate sample external data with SOM labels
        sample_external_data = pd.DataFrame(
            np.random.rand(1000, 10)
        )
        sample_external_data[self.cell_cc.som_col] = np.repeat(np.arange(1, 101), 10)

        labeled_external_data = self.cell_cc.assign_consensus_labels(sample_external_data)

        # ensure we've created a meta cluster column
        assert self.cell_cc.meta_col in sample_external_data.columns.values

        # ensure all the mappings match up
        external_mappings = labeled_external_data[
            [self.cell_cc.som_col, self.cell_cc.meta_col]
        ].copy()
        external_mappings = external_mappings.drop_duplicates().sort_values(
            by=self.cell_cc.som_col
        )
        true_mappings = self.cell_cc.mapping.sort_values(by=self.cell_cc.som_col)
        assert np.all(external_mappings.values == true_mappings.values)
