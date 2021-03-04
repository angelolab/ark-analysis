[![Build Status](https://travis-ci.com/angelolab/ark-analysis.svg?branch=master)](https://travis-ci.com/angelolab/ark-analysis)
[![Coverage Status](https://coveralls.io/repos/github/angelolab/ark-analysis/badge.svg?branch=master)](https://coveralls.io/github/angelolab/ark-analysis?branch=master)

# ark-analysis

Toolbox for analyzing multiplexed imaging data

Full documentation for the project can be found [here](https://ark-analysis.readthedocs.io/en/latest/)

## Info

This project contains code and example scripts for analyzing multiplexed imaging data

## To install the project:

Open terminal and navigate to where you want the code stored.

Then input the command:

```
$ git clone https://github.com/angelolab/ark-analysis.git
```

Next, you'll need to set up a docker image with all of the required dependencies.
 - First, [download](https://hub.docker.com/?overlay=onboarding) docker desktop. 
 - Once it's sucessfully installed, make sure it is running by looking in toolbar for the Docker whale.
 - Once it's running, enter the following commands into terminal 

```
$ cd ark-analysis
$ docker build -t ark-analysis .
``` 

You've now installed the code base. 

## Whenever you want to run the scripts:

Enter the following command into terminal from the same directory you ran the above commands:

```
$ bash start_docker.sh
``` 

This will generate a link to a jupyter notebook. Copy the last URL (the one with `127.0.0.1:8888` at the beginning) into your web browser.

Be sure to keep this terminal open.  **Do not exit the terminal or enter control-c until you are finished with the notebooks**.

### NOTE

If you already have a Jupyter session open when you run `$ bash start_docker.sh`, you will receive a couple additional prompts. 

Copy the URL listed after `Enter this URL instead to access the notebooks:` 

You will need to authenticate. Note the last URL (the one with `127.0.0.1:8888` at the beginning), copy the token that appears there (it will be after `token=` in the URL), paste it into the password prompt of the Jupyter notebook, and log in.

## Using the example notebooks:
- The Segment_Image_Data notebook walks you through the appropriate steps to format your data, run the data through deepcell, extracts the counts for each marker in each cell, and creats a csv file with the normalized counts
- The spatial_analysis notebook contains code for performing cluster- and channel-based randomization, as well as neighborhood analysis. 
- The example_visualization notebooks contains code for basic plotting functions and visualizations


## Once you are finished

You can shut down the notebooks and close docker by entering control-c in the terminal window.

## Updates

This project is still in development, and we are making frequent updates and improvements. If you want to update the version on your computer to have the latest changes, perform the following steps

First, get the latest version of the code

```
$ git pull
```

Then, run the command below to update the jupyter notebooks to the latest version
```
bash start_docker.sh --update
```
or
```
bash start_docker.sh -u
```

### WARNING

If you didn't change the name of any of the notebooks within the `scripts` folder, they will be overwritten by the command above!

If you have made changes to these notebooks that you would like to keep (specific file paths, settings, custom routines, etc), rename them before updating!

For example, rename your existing copy of `Segment_Image_Data.ipynb` to `Segment_Image_Data_old.ipynb`. Then, after running the update command, a new version of `Segment_Image_Data.ipynb` will be created with the newest code, and your old copy will exist with the new name that you gave it. 

After updating, you can copy over any important paths or modifications from the old notebooks into the new notebook

## Questions?

If you run into trouble, please first refer to our [FAQ](https://ark-analysis.readthedocs.io/en/latest/_rtd/faq.html). If that doesn't answer your question, you can open an [issue](https://github.com/angelolab/ark-analysis/issues). Before opening, please double check and see that someone else hasn't opened an issue for your question already. 

## Want to contribute?  

If you would like to help make `ark` better, please take a look at our [contributing guidelines](https://ark-analysis.readthedocs.io/en/latest/_rtd/contributing.html). 

## Citation
Please cite our paper if you found our repo useful!

[Greenwald, Miller et al. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning](https://www.biorxiv.org/content/10.1101/2021.03.01.431313v2)
