## FAQ

### When I run the notebook, it says "No module named 'ark'"

This error generally happens if you run the notebook outside of Docker. Simply running `jupyter notebook` or `jupyter lab` on a notebook will not start Docker for you. You need to explicitly run `bash start_docker.sh` and follow the prompts in our [README](https://github.com/angelolab/ark-analysis/blob/master/README.md). 

If you continue to run into issues, make sure you don't have any other Docker sessions running. 

### I got an error when running the notebook

The functions in `ark` have a lot of error checking built in. This means that if any of the arguments are wrong, you will get a ValueError of some kind. There's a good chance that the error message will tell you in very direct terms what the problem is, so please carefully read the error message. 

For example, if you get an invalid path error, the error message indicate which part of the path doesn't exist, helping you to troubleshoot. 

If you're still stuck, please completely close the notebook, kill the docker, and restart everything. If you've exhausted all of these options and are still getting the same error, feel free to open an [issue](https://github.com/angelolab/ark-analysis/issues/new/choose). 

### How can I help improve this project?

If you're interested in helping to add new features and develop this project, please check out our [contributing guidelines](https://ark-analysis.readthedocs.io/en/latest/_rtd/contributing.html)
