# Segmentation
Toolbox for segmenting multiplexed imaging data

To install the project to analyze your own data, open terminal and navigate to where you want the code stored.
Then:

```
$ git clone https://github.com/angelolab/segmentation.git
```

Next, you'll need to set up a docker image with all of the required dependencies. First, [download](https://hub.docker.com/?overlay=onboarding) docker desktop. 

Once it's sucessfully installed, make sure it is running by looking in toolbar for the Docker whale. Once it's running, enter the following code into terminal 

```
$ cd segmentation
$ docker build -t segmentation .
``` 

You've now installed the code base. Whenever you want to run the scripts, enter the following command into terminal from the same directory you ran the above commands:

```
$ bash start_docker.sh
``` 

This will generate a link to a jupyter notebook. Copy the last URL (the one with 127.0.0.1:8888 at the beginning) into your web browser.

To stop docker from running, enter control-c in the terminal window

The Jupyter Notebook has a deepcell-preprocessing and deepcell-postprocessing script. The preprocessing script is to generate the data that will be used for segmentation. The postprocessing script takes the output of deepcell and extracts single cell data from your imaging data. 
