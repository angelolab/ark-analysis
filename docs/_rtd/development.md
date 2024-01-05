## Information for Developers

If you wish to do higher-level development on top of `ark`, we recommend setting up a virtual environment. We highly recommend using `conda` virtual environments. To be able to set one up, you will need to install the Anaconda package.

### Setting up Your Virtual Environment - Anaconda

#### Installing Anaconda

For a step-by-step guide of how to install Anaconda, please refer to these links:
* https://docs.anaconda.com/anaconda/install/mac-os/ for Mac (x86_64 / Intel) users
* https://github.com/conda-forge/miniforge/releases for Mac (arm64 / Apple Silicon) users
* https://docs.anaconda.com/anaconda/install/windows/ for Windows users


**Notes for Mac users**

We recommend following the command line installer instructions as users have reported recent issues with the graphical installer. 
 
To test if `conda` has been added to your path, run `conda info` in your Terminal. If you get an error message, it means `conda` has not been added to your `PATH` environment variable yet. To fix, run `export PATH="/Users/yourname/anaconda3/bin:$PATH"`.

**Apple Silicon Installation**

You will need to install [*miniforge*](https://github.com/conda-forge/miniforge) first.
Miniforge contains conda with native Apple Silicon support. There are a few installation options available, all generally work the same way. Consult the documentation if you wish to read about them (using Mamba vs Conda for example).

1. Getting Miniforge
   * **Option 1: (recommended)** Install via homebrew
       ```sh
       brew install miniforge
       ```
   * **Option 2:** Download and Install via the terminal
        ```sh
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-$(uname)-$(uname -m).sh"
        ```
        ```sh
        bash Miniforge-$(uname)-$(uname -m).sh
        ```
2. Initialize it for shell
    ```sh
    conda init
    ```

#### Creating a virtual environment

Now that Anaconda is installed, you can now create a `conda` environment. 
 
To do so, on your command line, type `conda create -n <my_env> python=3.8`, where `<my_env>` is a name you set. Our codebase currently supports up to Python 3.8.
 
Say yes to any prompts and your `conda` environment will be created! 
 
To verify installation, activate your `conda` environment with `conda activate <my_env>`. If you see `(<my_env>)` on the far left of the command prompt, you have successfully created and activated your environment. Type `conda deactivate` to exit at any time.

### Setting up ark-analysis for development

`ark` relies on several other Python packages. Inside the `ark-analysis` repo (if you don't have it, first run `git clone https://github.com/angelolab/ark-analysis.git`), and with your virtual environment activated, you will need to install these other dependencies as well. Run `pip install -e ".[test]"` to install `ark` and it's dependencies and testing dependencies.
 
 
You're now set to start working with `ark-analysis`! Please look at [our contributing guidelines](contributing.html) for more information about development. For detailed explanations of the functions available to you in `ark`, please consult the Libraries section of this documentation. 

### Updating Ark Analysis in the Docker

**Note** that code changes aren't automatically propagated into the Docker Image.
However there may be times where you would like to work with and test out new changes and features.

You may update the current version of `ark-analysis` by running the following commands
in the Jupyter Lab terminal.

```sh
cd /opt/ark-analysis
git pull
pip install .
```

#### Using ark functions directly

If you will only be using functions in `ark` without developing on top of it, do not clone the repo. Simply run `pip install ark-analysis` inside the virtual environment to gain access to our functions. To verify installation, type `conda list ark-analysis` after completion. If `ark-analysis` is listed, the installation was successful. You can now access the `ark` library with `import ark`.

### Developing template notebooks via Docker

If you are using docker for your virtual environment, and plan to develop and commit template notebooks, then you should use the `--develop-notebook-templates` flag for `start_docker.sh`.

Typically, the `./templates` folder is copied into `./scripts` before starting docker and Jupyter is started within `./scripts`. This enables users of `ark-analysis` to use the notebooks without dirtying the git working directory—doing so would cause merge conflicts on pull. When using `--develop-notebook-templates`, `./templates` is used directly, so changes are changes reflected directly.

To enable, pass the either `-d` or  `--develop-notebook-templates` to `start_docker.sh`

    $ ./start_docker -d

Now notebooks can be `git diff`ed and `git commit`ed without having to copy changed notedbooks between `./scripts` and `./templates`.

### Building Docker Images Locally

It may be useful to be able to manually build a new Docker Image as features get added, changes made and libraries updated.
Specifically, updating Python libraries requires building a new docker image from scratch. 


Once you are in `ark-analysis`, the Docker Image can be built with the following command.
```
docker build -t ark-analysis .
``` 

The docker image will now build, and this process can take some time.

### More on xarrays

One type of N-D array we use frequently is `xarray` ([documentation](http://xarray.pydata.org/en/stable/)). The main advantages `xarray` offers are:

* Labeled dimension names
* Flexible indexing types

While these can be achieved in `numpy` to a certain extent, it's much less intuitive. In contrast, `xarray` makes it very easy to accomplish this. 

Just as `numpy`'s base array is `ndarray`, `xarray`'s base array is `DataArray`. We can initialize it with a `numpy` array as such (`xarray` should always be imported as `xr`):

```
arr = xr.DataArray(np.zeros((1024, 1024, 3)),
                   dims=['x', 'y', 'channel'],
                   coords=[np.arange(1024), np.arange(1024), ['red', 'green', 'blue']])
```

In this example, we assign the 0th, 1st, and 2nd dimensions to 'x', 'y', and 'channel' respectively. Both 'x' and 'y' are indexed with 0-1023, whereas 'channel' is indexed with RGB color names. 

Indexing for `xarray` works like `numpy`. For example, to extract an `xarray` with x=10:15, y=10:15, and channels=['red', 'blue']:

`arr.loc[10:15, 10:15, ['red', 'blue']]`

This can also be extracted into a `numpy` array using `.values`:

`arr.loc[10:15, 10:15, ['red', 'blue']].values`

Note the use of `.loc` in both cases. You do not have to use `.loc` to index, but you will be forced to use integer indexes. The following is equivalent to the above:

`arr[10:15, 10:15, [0, 2]].values`

In most cases, we recommend using `.loc` to get the full benefit of `xarray`. Note that this can also be used to assign values as well:

`arr.loc[10:15, 10:15, ['red', 'blue']] = 255`

To access the coordinate names, use `arr.dims`, and to access specific coordinate indices, use `arr.coord_name.values`. 

Finally, to save an `xarray` to a file, use:

`arr.to_netcdf(path, format="NETCDF3_64BIT")`

You can load the `xarray` back in using:

`arr = xr.load_dataarray(path)`


### Working with `AnnData`

We can load a single `AnnData` object using the function `anndata.read_zarr`, and several `AnnData` objects using the function `load_anndatas` from `ark.utils.data_utils`.

```python
from anndata import read_zarr
from ark.utils.data_utils import load_anndatas
```

```python
fov0 = read_zarr("data/example_dataset/fov0.zarr")
```

The channel intensities for each observation in the `AnnData` object with the `.to_df()` method, and get the channel names with `.var_names`.

```python
fov0.var_names
fov0.to_df()
```

The observations and their properties with the `obs` property of the `AnnData` object. The data here consists of measurements such as `area`, `perimeter`, and categorical information like `cell_meta_cluster` for each cell.

```python
fov0.obs
```

The $x$ and $y$ centroids of each cell can be accessed with the `obsm` attribute and the key `"spatial"`.

```python
fov0.obsm["spatial"]
```

We can load all the `AnnData` objects in a directory lazily with `load_anndatas`. We get a view of the `AnnData` objects in the directory.

```python
fovs_ac = load_anndatas(anndata_dir = "data/example_dataset/fov0.zarr")
```

We can utilize `AnnData` objects or `AnnCollections` in a similar way to a Pandas DataFrame. For example, we can filter the `AnnCollection` to only include cells that have a `cell_meta_cluster` label of `"CD4T"`.

```python
fovs_ac_cd4t = fovs_ac[fovs_ac.obs["cell_meta_cluster"] == "CD4T"]
print(type(fovs_ac_cd4t))
fovs_ac_cd4t.obs.df
```
The type of `fovs_ac_cd4t` is not an `AnnData` object, but instead an `AnnCollectionView`.
This is a `view` of the subset of the `AnnCollection`. This object can *only* access `.obs`, `.obsm`, `.layers` and `.X`.


We can subset a `AnnCollectionView` to only include the first $n$ observations objects with the following code. The slice based indexing behaves like a `numpy` array.

```python
n = 100
fovs_ac_cdt4_100 = fovs_ac_cd4t[:n]
fovs_ac_cd4t_100.obs.df
```

Often we will want to subset the `AnnCollection` to only include observations contained within a specific FOV.

```python
fov1_adata = fovs_ac[fovs_ac.obs["fov"] == "fov1"]

fov1_adata.obs.df
```

We can loop over all FOVs in a `AnnCollection` with the following code (there is alternative method in ):

```python
all_fovs = fovs_ac.obs["fov"].unique()

for fov in all_fovs:
    fov_adata = fovs_ac[fovs_ac.obs["fov"] == fov]
    # do something with fov_adata
```

Functions which take in `AnnData` objects can often be applied to `AnnCollections`.

The following works as expected:
```python
def dist(adata):
    x = adata.obsm["spatial"]["centroid_x"]
    y = adata.obsm["spatial"]["centroid_y"]
    return np.sqrt(x**2 + y**2)

dist(fovs_ac)
```

While the example below does not:
```python
from squidpy import gr
gr.spatial_neighbors(adata=fovs_ac, spatial_key="spatial")
```

This is due to a `AnnCollection` object not having a `uns` property.

#### Utilizing `DataLoaders`

While a `AnnCollection` can sometimes be used to apply a function over all FOVs, in some instances we either cannot do that, or perhaps we want to apply functions to each FOV independently.

We can access the underlying `AnnData` objects with `.adatas`.
```python
fovs_ac.adatas
```

In these instances we can construct data pipelines with [`torchdata`](https://pytorch.org/data/beta/index.html).


As an example, let's create a multi-stage `DataLoader` which does the following:
- Only extracts the observations with an area greater than `300`.
- Only extracts the observations which have a `cell_meta_cluster` label of `"CD4T"`.
- Compute the Spatial Neighbors graph for those observations (using [`squidpy.gr.spatial_neighbors`](https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors.html#squidpy.gr.spatial_neighbors)).


In order to construct a `torchdata` [`DataLoader2`](https://pytorch.org/data/beta/dataloader2.html) iterator we first need to create a `torchdata` [`IterDataPipe`](https://pytorch.org/data/beta/torchdata.datapipes.iter.html). This implements the `__iter__()` protocol, and represents an iterable over data samples.

We can convert the `AnnCollection` to a `torchdata` `IterDataPipe` with `ark.utils.data_utils.AnnDataIterDataPipe`.


```python
from ark.utils.data_utils import AnnDataIterDataPipe

fovs_ip = AnnDataIterDataPipe(fovs=fovs_ac)
```

The following two functions are used to filter the observations in the `AnnData` objects 
to only include cells with an area greater than `min_area` and cells with a `cell_meta_cluster` label of `cluster`.

```python
from anndata import AnnData

def filter_cells_by_cluster(adata: AnnData, cluster_label: str) -> AnnData:
    return adata[adata.obs["cell_meta_cluster"] == cluster_label]

def filter_cells_by_area(adata: AnnData, min_area: int) -> AnnData:
    return adata[adata.obs["area"] > min_area]
```

The following function is used to filter out `AnnData` objects which have no observations.
```python
def filter_empty_adata(fov: AnnData) -> bool:
    return len(fov) > 0
```


We can apply these functions to the `IterDataPipe` with the `map` and the `filter` method.
Because those methods return a new `IterDataPipe` object, we can chain them together.

```python
from functools import partial

cd4t_obs_filter = partial(filter_cells_by_cluster, cluster_label="CD4T")
area_obs_filter = partial(filter_cells_by_area, min_area=300)

fovs_subset = fovs_ip.map(cd4t_obs_filter).map(area_obs_filter).filter(filter_empty_adata)
```

The data pipeline can be visualized with `to_graph` function.

```python
from torchdata.datapipes.utils import to_graph

to_graph(fovs_subset)
```


The `DataLoader` can now be constructed.
```python
from torchdata.dataloader2.dataloader2 import DataLoader2

fovs_subset_dl = DataLoader2(fovs_subset)
```

We can now loop over the `DataLoader` and compute the Spatial Neighbors graph per FOV with the filtered observations.

```python
for fov in fovs_subset_dl:
    gr.spatial_neighbors(adata=fov, radius=350, spatial_key="spatial", coord_type="generic")
```

#### Further Reading
- [Official AnnData Documentation](https://anndata.readthedocs.io/en/latest/)
  - [Getting Started Tutorial](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html)
- [Converting from Single Cell Experiment and Seurat Objects](https://scanpy.readthedocs.io/en/stable/tutorials.html#conversion-anndata-singlecellexperiment-and-seurat-objects)
- [MuData - Multimodal AnnData](https://mudata.readthedocs.io/en/latest/index.html)
