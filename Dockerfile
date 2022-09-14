# Stage 1: Start from the official Python-3.8 image: https://hub.docker.com/_/python/
FROM python:3.8 AS base

# system maintenance
RUN apt-get update

# install dependencies needed for setting up R
RUN apt-get install -y lsb-release dirmngr gnupg apt-transport-https ca-certificates software-properties-common
RUN apt-get install -y libharfbuzz-dev libfribidi-dev
RUN apt-get install -y libcurl4-openssl-dev libssl-dev
RUN apt-get install -y libcurl4-gnutls-dev
RUN apt-get install -y libopenblas-base libgit2-dev

# add the correct Debian R repo
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7'
RUN add-apt-repository 'deb http://cloud.r-project.org/bin/linux/debian bullseye-cran40/'
# re-update based on previous setup
RUN apt-get update && apt -y upgrade

# install R
ARG rversion=4.0.4-1
RUN apt-get install -y \
    r-base-core=${rversion} \
    r-base-dev=${rversion} \
    r-base-html=${rversion} \
    r-doc-html=${rversion}

# install cmake (needed for nloptr)
RUN apt-get install -y cmake


# Stage 2: Adding R dependencies
FROM base AS r_deps
COPY --from=base . .
RUN R -e "install.packages('arrow')"
RUN R -e "install.packages('data.table')"
RUN R -e "install.packages('doParallel')"
RUN R -e "install.packages('foreach')"
RUN R -e "install.packages('BiocManager')"
RUN R -e "library(BiocManager); BiocManager::install('FlowSOM')"
RUN R -e "install.packages('devtools')"
RUN R -e "library(devtools); devtools::install_github('angelolab/FlowSOM')"
RUN R -e "BiocManager::install('ConsensusClusterPlus')"


# Stage 3: Adding ark-analysis (Python)
FROM r_deps AS ark_py
COPY --from=r_deps . .
# copy over the requirements.txt, install dependencies, and README
COPY setup.py pyproject.toml requirements.txt README.md /opt/ark-analysis/
RUN python -m pip install -r /opt/ark-analysis/requirements.txt

# copy the scripts over
# this should catch changes to the scripts from updates
COPY ark /opt/ark-analysis/ark

# Install the package via setup.py
RUN cd /opt/ark-analysis && python -m pip install .

# Stage 4: Copy Scripts, Jupyter Lab output
FROM ark_py as build_output
COPY --from=ark_py . .
WORKDIR /scripts

# jupyter lab
CMD jupyter lab --ip=0.0.0.0 --allow-root --no-browser --port=$JUPYTER_PORT --notebook-dir=/$JUPYTER_DIR
