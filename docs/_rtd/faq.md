## FAQ

### When I run the notebook, it says "No module named 'ark'"

This error almost universally happens if you run the notebook outside of Docker. Simply running `jupyter notebook` or `jupyter lab` on a notebook will not start Docker for you. You need to explicitly run `bash start_docker.sh` and follow the prompts in our [readme](landing.md). 

Even if you started Docker, make sure you actually copy the link into your browser and run our notebooks from there. Don't start another local Jupyter server on your computer to run the notebooks. 

### I found an unexpected bug

Double check the error message that is being printed. Errors often pop up due to malformed or inconsistent inputs that can be hard to catch (ex. missing channel names, different input dimensions). It often helps to have another look at the files/inputs you're giving the notebooks. 

Sometimes, restarting the notebook is necessary. Some of our functions (ex. `create_deepcell_output` in `Segment_Image_Data.ipynb`) cannot be run more than once in a given session. Other times, it's necessary to reset variables back to their default state. To do this, navigate to the `Kernel` tab in your notebook, and select `Restart Kernel...`. You can then run all the cells in a fresh environment. 

If you've followed the above steps and are still encountering the same bug, you can open an issue in our [issues](https://github.com/angelolab/ark-analysis/issues) page. Make sure that after you click `New issue` to select `Get started` next to `Bug report`. Please be as detailed as possible when describing the bug (your operating system, what you've tried so far, etc.). The more information we have the better! 

### I have a use case that you don't cover

Our repo is specifically for images scanned using MIBI. If you're attempting to use our image pipeline for non-MIBI images, you probably won't have much success. 

If you would like to do something slightly different with MIBI-scanned images, you can consider opening an enhancement issue in our [issues](https://github.com/angelolab/ark-analysis/issues) page. We can't guarantee we'll be able to handle all cases, but we examine every issue carefully and will let you know if it's feasible to add. Make sure that after you click `New issue` to select `Get started` next to `Enhancement`. 

If you would like to add a feature on your own, please check out our [contributing](contributing.md) guidelines. 

### I accidentally changed one of the source files

If that happens, don't panic. If you can't remember what the original code was, run `git restore path/to/file/changed` to reset the file back to its original state. 

For example, let's say you accidentally changed `segmentation_utils.py`. To restore the file back to its original state, simply run `git restore ark/utils/segmentation_utils.py`. The path must be computed from the `ark-analysis` root. 

A few exceptions to keep in mind:
* We have a few default images we keep in our `git` repo that you may end up overwriting. If that happens, you do not need to restore the original. 
* For Jupyter notebooks, we recommend that you make a copy of each notebook so you can keep track of what the original was. To do so, after you follow the steps for starting our Docker, click the checkbox next to the notebook you want to run, then click the `Duplicate` button at the top. A notebook with the format `NameOfNotebook-Copy-n.ipynb` will be created. Run that notebook instead to avoid any conflicts with the original. 
