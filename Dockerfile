# ANY CHANGES REQURIE A NEW RELEASE

# Stage 1: Start from the official Python 3.8 Image: https://hub.docker.com/_/python
FROM python:3.8 AS base

# system maintenance
RUN apt-get update && apt-get -y upgrade

# install gcc
RUN apt-get install -y gcc

# Stage 2: Installing Ark Analysis
FROM base AS move_ark

# copy over the requirements.txt, install dependencies, and README
COPY setup.py pyproject.toml requirements.txt README.md start_jupyter.sh /opt/ark-analysis/
RUN python -m pip install -r /opt/ark-analysis/requirements.txt


# Stage 3: Copy templates/ to scripts/
FROM move_ark AS move_templates

# copy the scripts over
# this should catch changes to the scripts from updates
COPY src/ /opt/ark-analysis/ark

# Stage 4: Install Ark Analysis
FROM move_templates AS install_ark

# Install the package via setup.py
RUN cd /opt/ark-analysis && python -m pip install .


# Stage 5: Set the working directory, and open Jupyter Lab
FROM install_ark AS open_for_user
WORKDIR /opt/ark-analysis

# jupyter lab
CMD bash start_jupyter.sh
