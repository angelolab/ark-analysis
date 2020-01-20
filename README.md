# Segmentation
Analysis of MIBI segmentation accuracy

Useful mibi i/o functions are in the utils folders. To add to your project:

```
pip install git+git://github.com/angelolab/segmentation.git

from segmentation import utils
```

To install the project to analyze your own data, open terminal and navigate to where you want the code stored.
Then:

```
$ git clone https://github.com/angelolab/segmentation.git
```

Next, you'll need to set up a docker image with all of the required dependencies. First, go to https://hub.docker.com/?overlay=onboarding and download docker desktop. 

Once it's sucessfully installed, make sure it is running by looking in toolbar for the Docker whale. Once it's running, enter the following code into terminal 

```
$ cd segmentation
$ docker build -t $USER/segmentation .
$ bash start_docker.sh
``` 

Copy the last URL (the one with 127.0.0.1:8888 at the beginning) into your web browser. This will take you to the folder with jupyter notebooks to run the code


To run the pipeline on your own data, just copy the relevant files in the /data directory.
