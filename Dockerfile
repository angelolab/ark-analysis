# FROM python:3.6
FROM ubuntu:16.04

# system maintenance
RUN apt-get update

# install dependencies needed for setting up R
# RUN apt-get install -y lsb-release dirmngr gnupg apt-transport-https ca-certificates software-properties-common

# install the package needed for add-apt-repository
RUN apt-get update && apt-get install -y software-properties-common apt-transport-https ca-certificates gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt-get install -y apt-transport-https ca-certificates software-properties-common

# get the Linux distro info, use this to set the right R download
# RUN lsb_release -a

# install Python
RUN apt-get update && apt-get install -y python3.6 python3-distutils python3-pip python3-apt

# set up the key for adding the R repo
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9

# add the correct Linux R repo
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran40/'

# ensure the previous command added the right line to the sources.list files
RUN cat /etc/apt/sources.list

# re-update based on previous setup
RUN apt-get update && apt-get -y upgrade

# install gcc and R
RUN apt-get install -y --allow-unauthenticated gcc r-base

WORKDIR /scripts

# copy over the requirements.txt, install dependencies, and README
COPY setup.py requirements.txt README.md /opt/ark-analysis/
COPY .toks /home/.toks
RUN pip install -r /opt/ark-analysis/requirements.txt

# copy the scripts over
COPY ark /opt/ark-analysis/ark

# Install the package via setup.py
RUN pip install /opt/ark-analysis

# Install R dependency packages
RUN R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/BH/BH_1.75.0-0.tar.gz', repos=NULL, type='source')"
RUN R -e "install.packages('arrow')"
RUN R -e "install.packages('data.table')"
RUN R -e "install.packages('BiocManager')"
RUN R -e "BiocManager::install('FlowSOM')"
RUN R -e "BiocManager::install('ConsensusClusterPlus')"

# jupyter lab
CMD jupyter lab --ip=0.0.0.0 --allow-root --no-browser --port=$JUPYTER_PORT --notebook-dir=/$JUPYTER_DIR
