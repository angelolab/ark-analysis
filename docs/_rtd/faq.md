## FAQ

### When I run the notebook, it says "No module named 'ark'"

This error generally happens if you run the notebook outside of Docker. Simply running `jupyter notebook` or `jupyter lab` on a notebook will not start Docker for you. You need to explicitly run `bash start_docker.sh` and follow the prompts in our [README](https://github.com/angelolab/ark-analysis/blob/main/README.md). 

If you continue to run into issues, make sure you don't have any other Docker sessions running. 

### I got an error when running the notebook

The functions in `ark` have a lot of error checking built in. This means that if any of the arguments are wrong, you will get a ValueError of some kind. There's a good chance that the error message will tell you in very direct terms what the problem is, so please carefully read the error message. 

For example, if you get an invalid path error, the error message indicate which part of the path doesn't exist, helping you to troubleshoot. 

If you're still stuck, please completely close the notebook, kill the docker, and restart everything. If you've exhausted all of these options and are still getting the same error, feel free to open an [issue](https://github.com/angelolab/ark-analysis/issues/new/choose). 

### My kernel keeps dying

This means Docker is running out of memory. To increase the memory, open Docker Preferences (on Mac, click the whale logo with boxes on the top menu bar, and select Preferences). Select Resources on the left panel, and a slider will appear for Docker memory usage which you can use to increase. 

Keep in mind that you should be careful increasing your Docker memory usage above half of your computer RAM. For example, if you have a 16 GB computer, we recommend not increasing your Docker memory above 8 GB. 

Avoid adjusting the CPUs, Swap, and Disk Image Size sliders. 

### How can I help improve this project?

If you're interested in helping to add new features and develop this project, please check out our [contributing guidelines](https://ark-analysis.readthedocs.io/en/latest/_rtd/contributing.html). 

### I have a different question

Feel free to check out our [issues](https://github.com/angelolab/ark-analysis/issues) page. If someone doesn't already have an open issue for your question, feel free to open one yourself.
