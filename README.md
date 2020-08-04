# Segmentation
Toolbox for segmenting multiplexed imaging data

## Info

This project contains a deepcell-preprocessing and deepcell-postprocessing notebook. The preprocessing notebook reformats imaging-data for deepcell's 'Multiplex' segmentation. The postprocessing notebook takes the output of deepcell and consolidates your imaging data into an xarray.

## To install the project:

Open terminal and navigate to where you want the code stored.

Then input the command:

```
$ git clone https://github.com/angelolab/segmentation.git
```

Next, you'll need to set up a docker image with all of the required dependencies.
 - First, [download](https://hub.docker.com/?overlay=onboarding) docker desktop. 
 - Once it's sucessfully installed, make sure it is running by looking in toolbar for the Docker whale.
 - Once it's running, enter the following commands into terminal 

```
$ cd segmentation
$ docker build -t segmentation .
``` 

You've now installed the code base. 

## Whenever you want to run the scripts:

Enter the following command into terminal from the same directory you ran the above commands:

```
$ bash start_docker.sh
``` 

This will generate a link to a jupyter notebook. Copy the last URL (the one with 127.0.0.1:8888 at the beginning) into your web browser.

Be sure to keep this terminal open.  **Do not exit the terminal or enter control-c until you are finished with the notebooks**.

## Once you are finished

You can shut down the notebooks and close docker by entering control-c in the terminal window.

## Updates

You can update your notebooks using the command:

```
$ git pull
```
