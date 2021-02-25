FROM python:3.6

# system maintenance
RUN apt-get update && apt-get install -y gcc r-base

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
RUN R -e "BiocManager::install('ConsensusClusterPlus')"

# jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
