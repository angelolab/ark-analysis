# ANY CHANGES REQURIE A NEW RELEASE

FROM python:3.8

# system maintenance
RUN apt-get update && apt-get -y upgrade

# install gcc
RUN apt-get install -y gcc

# # install cmake (needed for nloptr)
# RUN apt-get install -y cmake

# Install ark-analysis
# copy over the requirements.txt, install dependencies, and README
COPY setup.py pyproject.toml requirements.txt README.md start_jupyter.sh /opt/ark-analysis/
RUN python -m pip install -r /opt/ark-analysis/requirements.txt

# copy the scripts over
# this should catch changes to the scripts from updates
COPY ark /opt/ark-analysis/ark

# Install the package via setup.py
RUN cd /opt/ark-analysis && python -m pip install .

WORKDIR /opt/ark-analysis

# jupyter lab
CMD bash start_jupyter.sh
