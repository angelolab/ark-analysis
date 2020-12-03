## Setting Up Your Virtual Environment

If you wish to do higher-level development on top of `ark`, we recommend setting up a virtual environment. We highly recommend using `conda` virtual environments. To be able to set one up, you will need to install the Anaconda package.

### Installing Anaconda

For a step-by-step guide of how to install Anaconda, please refer to these links:
* https://docs.anaconda.com/anaconda/install/mac-os/ for Mac users
* https://docs.anaconda.com/anaconda/install/windows/ for Windows users

#### Notes for Mac users

We recommend following the command line installer instructions as users have reported recent issues with the graphical installer. 
 
To test if `conda` has been added to your path, run `conda info` in your Terminal. If you get an error message, it means `conda` has not been added to your `PATH` environment variable yet. To fix, run `export PATH="/Users/yourname/anaconda3/bin:$PATH"`.

### Creating a virtual environment

Now that Anaconda is installed, you can now create a `conda` environment. 
 
To do so, on your command line, type `conda create -n <my_env> python=3.6`, where `<my_env>` is a name you set. Our codebase only supports Python 3.6, so please do not change the `python=3.6` flag when creating your environment. 
 
Say yes to any prompts and your `conda` environment will be created! 
 
To verify installation, activate your `conda` environment with `conda activate <my_env>`. If you see `(<my_env>)` on the far left of the command prompt, you have successfully created and activated your environment. Type `conda deactivate` to exit at any time.

### Setting up ark-analysis for development

`ark` relies on several other Python packages. Inside the `ark-analysis` repo (if you don't have it, first run `git clone https://github.com/angelolab/ark-analysis.git`), and with your virtual environment activated, you will need to install these other dependencies as well. Run `pip install -r requirements.txt` to do so. 
 
Note that you will not have access to `ark` or the other libraries inside `requirements.txt` outside of this virtual environment. 
 
You're now set to start working with `ark-analysis`! Please look at [our contributing guidelines](contributing.html) for more information about development. For detailed explanations of the functions available to you in `ark`, please consult the Libraries section of this documentation. 

### Using ark functions directly

If you will only be using functions in `ark` without developing on top of it, do not clone the repo. Simply run `pip install ark-analysis` inside the virtual environment to gain access to our functions. To verify installation, type `conda list ark-analysis` after completion. If `ark-analysis` is listed, the installation was successful. You can now access the `ark` library with `import ark`.

### More on xarrays

One type of N-D array with frequently is `xarray` ([documentation]http://xarray.pydata.org/en/stable/). The main advantages `xarray` offers are:

* Labeled dimension names
* Flexible indexing types

While these can be achieved in `numpy` to a certain extent, it is much less intuitive. In contrast, `xarray` makes it very easy to accomplish this. 

Just as `numpy`'s base array is `ndarray`, `xarray`'s base array is `DataArray`. We can initialize it with a numpy array as such (`xarray` should always be imported as `xr`):

```
arr = xr.DataArray(np.zeros((1024, 1024, 3)),
                   dims=['x', 'y', 'channel'],
                   coords=[np.arange(1024), np.arange(1024), ['red', 'green', 'blue']])
```

In this example, we assign the 0th, 1st, and 2nd dimensions to names 'x', 'y', and 'channel'. Both 'x' and 'y' are indexed with 0-1023, whereas 'channel' is indexed with RGB color names. 

Indexing works much like `numpy` arrays. For example, if I want to extract an `xarray` subset on just x and y in range 10:15 and the red and blue channels:

`arr.loc[10:15, 10:15, ['red', 'blue']]`

I can also extract these values into a `numpy` array using `.values`:

`arr.loc[10:15, 10:15, ['red', 'blue']].values`

Note the use of `.loc` in both cases. You do not have to use `.loc` to index, but you will be forced to use only integer indexes otherwise. The following is equivalent to the above:

`arr[10:15, 10:15, [0, 2]].values`

In most cases, we recommend using `.loc` to get the full benefit of an `xarray`. Note that this can also be used to assign values as well:

`arr.loc[10:15, 10:15, ['red', 'blue']] = 255`

To access the coordinate names, use `arr.dims`, and to access the specific coordinate values, use `arr.coord_name.values`. 

Finally, to save an `xarray` to a file, use:

`arr.to_netcdf(path, format="NETCDF3_64BIT")`

You can load the `xarray` back in using:

`arr = xr.load_dataarray(path)`
