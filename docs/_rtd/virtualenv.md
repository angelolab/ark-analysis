## Setting Up Your Virtual Environment

If you wish to do higher-level development on top of `ark`, we recommend setting up a virtual environment. We highly recommend using `conda` virtual environments. To be able to set one up, you will need to install the Anaconda package.

### Installing Anaconda

For a step-by-step guide of how to install Anaconda, please refer to these links:
* https://docs.anaconda.com/anaconda/install/mac-os/ for Mac users
* https://docs.anaconda.com/anaconda/install/windows/ for Windows users

We recommend using the graphical installer for ease of use.

### Creating a virtual environment

Now that Anaconda is installed, you can now create a `conda` environment. 
 
To do so, on your command line, type `conda create -n <my_env> python=3.6`, where `<my_env>` is a name you set. Our codebase only supports Python 3.6, so please do not change the `python=3.6` flag when creating your environment. 
 
Say yes to any prompts and your `conda` environment will be created! 
 
To verify installation, activate your `conda` environment with `conda activate <my_env>`. If you see `(<my_env>)` on the far left of the command prompt, you have successfully created and activated your environment. Type `conda deactivate` to exit at any time.

### Setting up ark-analysis for development

`ark` relies on several other Python packages. Inside the `ark-analysis` repo (if you don't have it, first run `git clone https://github.com/angelolab/ark-analysis.git`), and with your virtual environment activated, you will need to install these other dependencies as well. Run `pip install -r requirements.txt` to do so. 
 
Note that you will not have access to `ark` or the other libraries inside `requirements.txt` outside of this virtual environment. 
 
You're now set to start working with `ark-analysis`! Please look at [our contributing guidelines](contributing.md) for more information about development. For detailed explanations of the functions available to you in `ark`, please consult the Libraries section of this documentation. 

### Using ark functions directly

If you will only be using functions in `ark` without developing on top of it, do not clone the repo. Simply run `pip install ark-analysis` to gain access to our functions. To verify installation, type `conda list ark-analysis` after completion. If `ark-analysis` is listed, the installation was successful. You can now access the `ark` library by running `import ark`.
 
