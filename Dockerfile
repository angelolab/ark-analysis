# ANY CHANGES REQURIE A NEW RELEASE

# Stage 1: Start from the official Python 3.8 Image: https://hub.docker.com/_/python
FROM python:3.8 AS base

# Set environment variable 
ENV RUNNING_IN_DOCKER true

# system maintenance
RUN apt update && apt -y upgrade

# install gcc
RUN apt-get install -y gcc

# Install zsh shell, default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -t robbyrussell

# Stage 2: Installing Ark Analysis
FROM base AS move_ark

# copy over: setup.py, pyproject.toml, README and start_jupyter.sh script
COPY setup.py pyproject.toml README.md start_jupyter.sh /opt/ark-analysis/

# Stage 3: Copy templates/ to scripts/
FROM move_ark AS move_templates

# copy the scripts over
# this should catch changes to the scripts from updates
COPY src /opt/ark-analysis/src

# Stage 4: Install Ark Analysis
FROM move_templates AS install_ark

# Install the package via setup.py
RUN cd /opt/ark-analysis && python -m pip install .

# Stage 5: Set the working directory, and open Jupyter Lab
FROM install_ark AS open_for_user
WORKDIR /opt/ark-analysis

# jupyter lab
CMD bash start_jupyter.sh
