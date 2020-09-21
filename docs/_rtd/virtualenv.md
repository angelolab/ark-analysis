## Setting Up Your Virtual Environment

If you wish to do higher-level development on top of `ark`, we recommend setting up a virtual environment. We highly recommend using `conda` virtual environments. To be able ot set one up, you will need to install the Anaconda package.

### Installing Anaconda

For a step-by-step guide of how to install Anaconda, please refer to these links:
* https://docs.anaconda.com/anaconda/install/mac-os/ for Mac users
* https://docs.anaconda.com/anaconda/install/windows/ for Windows users

We recommend using the graphical installer for ease of use.

### Creating a virtual environment

Now that Anaconda is installed, you can now create a `conda` environment. 
 
To do so, on your command line, type `conda create -n <my_env> python=3.6`, where `<my_env>` is a name you set. We highly recommend setting `python=3.6` because this is the version of Python our code uses. 
 
Say yes to any prompts and your `conda` environment will be created! 
 
To verify installation, activate your `conda` environment with `conda activate <my_env>`. If you see `(<my_env>)` on the far left of the command prompt, you have successfully created and activated your environment. Type `conda deactivate` to exit at any time.

### Installing ark

Inside your `conda` environment, you will need to run `pip install ark-analysis` to gain access to our repo. To verify installation, type `conda list ark-analysis` after completion. If `ark-analysis` is listed, the installation was successful. You can now access the `ark` library. For detailed explanations of the functions available to you, please look at the `Library` section of this documentation. 
 
Note that you will not have access to `ark` outside of this virtual environment. 

