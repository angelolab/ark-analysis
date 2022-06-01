FROM python:3.6

# system maintenance
RUN apt-get update

# install dependencies needed for setting up R
RUN apt-get install -y lsb-release dirmngr gnupg apt-transport-https ca-certificates software-properties-common

# set up the key for adding the R repo
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7

# add the correct Linux R repo
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/debian bullseye-cran40/'

# re-update based on previous setup
RUN apt-get update && apt-get -y upgrade

# install gcc and R
RUN apt-get install -y gcc r-base

# install cmake (needed for nloptr)
RUN apt-get install -y cmake

WORKDIR /scripts

# copy over the requirements.txt, install dependencies, and README
COPY setup.py requirements.txt README.md /opt/ark-analysis/
RUN pip install -r /opt/ark-analysis/requirements.txt

# copy the scripts over
COPY ark /opt/ark-analysis/ark

# Install the package via setup.py
RUN pip install /opt/ark-analysis

# Install R dependency packages
RUN R -e "install.packages('arrow')"
RUN R -e "install.packages('data.table')"
RUN R -e "install.packages('BiocManager')"
RUN R -e "BiocManager::install('FlowSOM')"
RUN R -e "install.packages('devtools')"
RUN R -e "library(devtools); devtools::install_github('angelolab/FlowSOM')" # this ensures we retrieve the forked FlowSOM
RUN R -e "BiocManager::install('ConsensusClusterPlus')"

# jupyter lab
CMD jupyter lab --ip=0.0.0.0 --allow-root --no-browser --port=$JUPYTER_PORT --notebook-dir=/$JUPYTER_DIR
