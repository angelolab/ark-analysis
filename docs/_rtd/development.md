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

**Apple Silicon Installation**

`Ark` now supports development with Apple Silicon.

Currently there is not a native M1 implementation of ark-analysis for development, so it will need to go through Rosetta 2 (the Intel to Arm transition layer). Luckily, this isn't something you'll have to deal with, as `conda` makes it straightforward.

1. Create a Python 3.8 `conda` environment called `my_env`.
    ```sh
    conda create -n <my_env> python=3.8
    ```

2. Test to make sure the `platform.machine()` function reports `arm64` in the terminal.
    ```sh
    conda activate <my_env>
    python -c "import platform;print(platform.machine())"
    ```

3. Force `conda` commands to use Intel Mac packages.
    ```
    conda config --env --set subdir osx-64
    ```
4. The prompt may ask you to deactivate and reactivate the environment as well.

Now any package that is installed in `my_env` will targeted for `arm64`.

### Setting up ark-analysis for development

`ark` relies on several other Python packages. Inside the `ark-analysis` repo (if you don't have it, first run `git clone https://github.com/angelolab/ark-analysis.git`), and with your virtual environment activated, you will need to install these other dependencies as well. Run `pip install -r requirements.txt` to do so. 
 
Note that you will not have access to `ark` or the other libraries inside `requirements.txt` outside of this virtual environment. 
 
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

It may be useful to be able to manually build a new Docker Image as features get added, changes made and libraries updated. This
will allow you to test and experience bleeding edge changes, as they can't necessarily be adjusted in the `requirements.txt` file.
Specifically, updating Python libraries requires building a new docker image from scratch. 


Once you are in `ark-analysis`, the Docker Image can be built with the following command.
```
docker build -t ark-analysis .
``` 

The docker image will now build and this process can take some time.


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


### Creating a New Release:

There are several steps for creating a new Release of `Ark`. 
The versioning format is:

```
MAJOR.MINOR.PATCH
```

For example, suppose that the current version is `A.B.C`, and we need to create a new release `X.Y.Z`. The following instructions describe this procedure.

**Create a new PR with the following format as the branch name:**

```
next_release_vX.Y.Z
```
**In that branch:**
1. Set the label for the PR to `dependencies`.
1. Bump the `VERSION` Variable in `setup.py` to `X.Y.Z`. View the [draft release notes](https://github.com/angelolab/ark-analysis/releases) to read the current bugfixes, enhancements and more.
   1. If, in the release notes draft there are PRs that are not categorized, label them appropriately (usually based on the label of their respective Issue).
2. Make sure that all tests pass for `Ark` on Travis-CI. 
3. In the `ark-analysis/start_docker.sh` script, change the `VERSION` variable from `vA.B.C` to `vX.Y.Z`
4. Modify the `.travis.yml` CI configuration script to allow `test_pypi_deploy` to run (comment the line `if: tag IS present`).
5. Request a review and merge the `Ark` branch.
6. Next head to the most recent Drafted Release Notes:
   1. Double check that the tag is the appropriate version name.
   2. Publish the Release.
   3. Next the `Ark` will be pushed to PyPI and the Docker Image will be built on Travis CI. 

**Test Changes on Toffy**
1. Test the effects that changes in `Ark` have on `toffy` locally.
   1. Install the new branch of `Ark` in your Python environment with 
       ```
       pip install -e <location/to/ark>
       ```
   2. **As needed**, sync with `toffy` and `mibi-bin-tools`
      1. Update relevant packages in these repos, such as `scikit-image` or `xarray`.
      2. Locally, test that the new version works with `toffy`
      3. If there are errors in `toffy` fix them in a separate branch named:
         ```
         ark_vX.Y.Z_compatibility
         ```
   3. If necessary, change the version of ark-analysis in `toffy/requirements.txt`:
      ```
      git+https://github.com/angelolab/ark-analysis.git@vA.B.C -> git+https://github.com/angelolab/ark-analysis.git@vX.Y.Z
      ```

2. Once all errors have been ironed out create PRs for the respective changes in the effected repositories, and label them as `dependencies`.
3. Merge the compatibility PRs.
