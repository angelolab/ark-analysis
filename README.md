# Pixie

Pipeline for pixel clustering and cell clustering of multiplexed imaging data as described in [Liu et al. Robust phenotyping of highly multiplexed tissue imaging data using pixel-level clustering](https://doi.org/10.1101/2022.08.16.504171).

**For a maintained version of the codebase, please see https://github.com/angelolab/ark-analysis.** The codebase in this repository reflects a static version of the pipeline as described in Liu et al., but is not actively maintained.

## Table of Contents
- [Getting Started](#getting-started)
  - [Overview](#overview)
    - [1. Pixel clustering with Pixie](#1-pixel-clustering-with-pixie)
    - [2. Cell clustering with Pixie](#2-cell-clustering-with-pixie)
  - [Installation Steps](#installation-steps)
    - [Download the Repo](#download-the-repo)
    - [Setting up Docker](#setting-up-docker)
    - [Using the Repository (Running the Docker)](#using-the-repository-running-the-docker)
    - [Pip](#pip)
    - [Running on Windows](#running-on-windows)
- [Example Dataset](#example-dataset)
- [Questions?](#questions)


## Getting Started

### Overview
This repo contains a pipeline for pixel clustering and cell clustering of multiplexed imaging data. The assumption is that you've already performed any necessary image processing on your data (such as denoising, background subtraction, autofluorescence correction, etc), and that it is ready to be analyzed. 

#### 1. Pixel clustering with Pixie  
The first step in the [Pixie](https://www.biorxiv.org/content/10.1101/2022.08.16.504171v1) pipeline is to run the [**pixel clustering notebook**](./templates/2_Pixie_Cluster_Pixels.ipynb). The notebook walks you through the process of generating pixel clusters for your data, and lets you specify what markers to use for the clustering, train a model, use it to classify your entire dataset, and generate pixel cluster overlays. The notebook includes a GUI for manual cluster adjustment and annotation. [Workshop Talk - Pixel Level Analysis](https://youtu.be/e7C1NvaPLaY)

#### 2. Cell clustering with Pixie  
The second step in the [Pixie](https://www.biorxiv.org/content/10.1101/2022.08.16.504171v1) pipeline is to run the [**cell clustering notebook**](./templates/3_Pixie_Cluster_Cells.ipynb). This notebook will use the pixel clusters generated in the first notebook to cluster the cells in your dataset. The notebook walks you through generating cell clusters for your data and generates cell cluster overlays. The notebook includes a GUI for manual cluster adjustment and annotation. [Workshop Talk - Cell-level Analysis - Part 2: Cell Clustering](https://youtu.be/4_AJxrxPYlk?t=2704)


### Installation Steps

#### Download the Repo

Open terminal and navigate to where you want the code stored. Clone the repo:

```
git clone https://github.com/angelolab/pixie.git
```

As this is an ongoing project, extra features may be added to the Pixie pipeline in the future. For an actively maintained version of the pipeline, please see https://github.com/angelolab/ark-analysis. The codebase in this `pixie` repo corresponds to `v0.6.0` of `ark`.

#### Setting up Docker

For ease-of-use, we have created a Docker container to run this pipeline. Docker is a containerization platform that allows programs to be packaged into containers, which are standardized executable components that combine source code with OS libraries and dependences needed to run that code. We have created a [setup video](https://youtu.be/EXMGdi_Izdw).

Next, you'll need to download Docker Desktop:
 - First, [download](https://hub.docker.com/?overlay=onboarding) Docker Desktop. 
 - Once it's sucessfully installed, make sure it is running by looking in toolbar for the Docker whale icon. 


#### Using the Repository (Running the Docker)

Enter the following command into terminal from the same directory you ran the above commands:

```
./start_docker.sh
``` 

If running for the first time, or if our Docker image has updated, it may take a while to build and setup before completion. 

This will generate a link to a Jupyter notebook. Copy the last URL (the one with `127.0.0.1:8888` at the beginning) into your web browser. Be sure to keep this terminal open.  **Do not exit the terminal or enter `control-c` until you are finished with the notebooks**. 

**NOTE:** If you already have a Jupyter session open when you run `./start_docker.sh`, you will receive a couple additional prompts. Copy the URL listed after `Enter this URL instead to access the notebooks:`. You will need to authenticate. Note the last URL (the one with `127.0.0.1:8888` at the beginning), copy the token that appears there (it will be after `token=` in the URL), paste it into the password prompt of the Jupyter notebook, and log in.

You can shut down the notebooks and close docker by entering `control-c` in the terminal window.

**Remember to duplicate and rename notebooks.** If you didn't change the name of any of the notebooks within the `templates` folder, they will be overwritten when you decide to update the repo.


#### Pip

While we recommend users to use docker since it takes care of dependencies in different envrionments, if you choose to use this pipeline outside of docker, this pipeline can be installed using pip:

```
pip install ark-analysis==0.6.0
```

Note that this pipeline requires python 3.8.


#### Running on Windows

Our repo runs best on Linux-based systems (including MacOS). If you need to run on Windows, please consult our [Windows guide](https://ark-analysis.readthedocs.io/en/latest/_rtd/windows_setup.html) for additional instructions.

## Example Dataset

If you would like to test out the pipeline, then we have incorporated an example MIBI-TOF dataset within the notebooks. The dataset contains 11 FOVs with 22 channels (CD3, CD4, CD8, CD14, CD20, CD31, CD45, CD68, CD163, CK17, Collagen1, ECAD, Fibronectin, GLUT1, H3K9ac, H3K27me3, HLADR, IDO, Ki67, PD1, SMA, Vim), and intermediate data necessary for each notebook in the pipeline.

We utilize [**Hugging Face**](https://huggingface.co) for storing the dataset and using their API's for creating these configurations. You can view the [dataset](https://huggingface.co/datasets/angelolab/ark_example)'s repository as well.

### Dataset Compartments

**Image Data:** This compartment stores the tiff files for each channel, for every FOV.
```sh
image_data/
├── fov0/
│  ├── CD3.tiff
│  ├── ...
│  └── Vim.tiff
├── fov1/
│  ├── CD3.tiff
│  ├── ...
│  └── Vim.tiff
├── .../
```

**Cell Table:** This compartment stores example cell tables.

```sh
segmentation/cell_table/
├── cell_table_arcsinh_transformed.csv
├── cell_table_size_normalized.csv
└── cell_table_size_normalized_cell_labels.csv
```

**Deepcell Output:** This compartment stores example segmentation images after running segmentation using Deepcell.
```sh
segmentation/deepcell_output/
├── fov0_whole_cell.tiff
├── fov0_nuclear.tiff
├── ...
├── fov10_whole_cell.tiff
└── fov10_nuclear.tiff
```

**Example Pixel Output:** This compartment stores feather files, csvs and pixel masks generated by pixel clustering.

```sh
segmentation/example_pixel_output_dir/
├── cell_clustering_params.json
├── channel_norm.feather
├── channel_norm_post_rowsum.feather
├── pixel_thresh.feather
├── pixel_channel_avg_meta_cluster.csv
├── pixel_channel_avg_som_cluster.csv
├── pixel_masks/
│  ├── fov0_pixel_mask.tiff
│  └── fov1_pixel_mask.tiff
├── pixel_mat_data/
│  ├── fov0.feather
│  ├── ...
│  └── fov10.feather
├── pixel_mat_subset/
│  ├── fov0.feather
│  ├── ...
│  └── fov10.feather
├── pixel_meta_cluster_mapping.csv
└── pixel_som_weights.feather
```

**Example Cell Output:** This compartment stores feather files, csvs and cell masks generated by cell clustering.

```sh
segmentation/example_cell_output_dir/
├── cell_masks/
│  ├── fov0_cell_mask.tiff
│  └── fov1_cell_mask.tiff
├── cell_meta_cluster_channel_avg.csv
├── cell_meta_cluster_count_avg.csv
├── cell_meta_cluster_mapping.csv
├── cell_som_cluster_channel_avg.csv
├── cell_som_cluster_count_avg.csv
├── cell_som_weights.feather
├── cluster_counts.feather
├── cluster_counts_size_norm.feather
└── weighted_cell_channel.csv
```


## Questions?

If you have a general question or are having trouble with part of the repo, please see https://github.com/angelolab/ark-analysis. You can refer to our [FAQ](https://ark-analysis.readthedocs.io/en/latest/_rtd/faq.html) or head to the [discussions](https://github.com/angelolab/ark-analysis/discussions) tab to get help. If you've found a bug with the codebase, first make sure there's not already an [open issue](https://github.com/angelolab/ark-analysis/issues), and if not, you can then [open an issue](https://github.com/angelolab/ark-analysis/issues/new/choose) describing the bug.
