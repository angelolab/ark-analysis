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
