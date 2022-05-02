FROM python:3.6

# system maintenance
RUN apt-get update
RUN apt-get install -y dirmngr gnupg apt-transport-https ca-certificates software-properties-common
RUN apt-get install libcurl3 libjpeg8 libpng12 libreadline6
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran40/'
RUN cat /etc/apt/sources.list
RUN cat /etc/apt/sources.list.d
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y gcc r-base

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
