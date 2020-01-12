# Segmentation
Analysis of MIBI segmentation accuracy

Useful mibi i/o functions are in the utils folders. To add to your project:

```
pip install git+git://github.com/angelolab/segmentation.git

from segmentation import utils
```

To install the project for local use, open terminal and navigate to where you want the code stored.
Then:

```
$ git clone https://github.com/angelolab/segmentation.git
```

Next, you'll need to set up a virtual environment with the code

```
$ cd segmentation
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ ipython kernel install --user --name=segmentation
$ deactivate
``` 

Now you'll set up jupyter notebooks to run the code

```
$ pip install jupyter
```

To run the pipeline, open up the relevant notebook in the scripts folder:

```
$ jupyter notebook scripts/deepcell_postprocessing.ipynb
```
