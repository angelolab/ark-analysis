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

This will generate a link to a jupyter notebook. Copy the last URL (the one with 127.0.0.1:8888 at the beginning) into your web browser.

Be sure to keep this terminal open.  **Do not exit the terminal or enter control-c until you are finished with the notebooks**.

## Using the example notebooks:
- The Deepcell_preprocessing notebook walks you through the appropriate formatting steps in order to run your data through DeepCell to be segmented
- The Deepcell_postprocessing notebooks takes the segmentation predictions from DeepCell, and uses them to extract the counts of each marker from your dataset
- The spatial_analysis notebook contains code for performing cluster- and channel-based randomization, as well as neighborhood analysis. 


## Once you are finished

You can shut down the notebooks and close docker by entering control-c in the terminal window.

## Updates

This project is still in development, and we are making frequent updates and improvements. If you want to update the version on your computer to have the latest changes, perform the following steps

First, copy the latest version of the code

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

Examples:
```
Deepcell_Postprocessing.ipynb -> old_postprocessing.ipynb
                              or postprocessing_saved.ipynb
                              or old_dcpostP.ipynb
                              ...
```

After updating, you can bring changes from the old notebooks into the new notebooks at your discretion
