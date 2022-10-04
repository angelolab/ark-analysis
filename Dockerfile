# Stage 1: Start from the official Python-3.8 image: https://hub.docker.com/_/python/
FROM python:3.8 AS base

# system maintenance
RUN apt-get update

# install dependencies needed for setting up R
RUN apt-get install -y lsb-release dirmngr gnupg apt-transport-https ca-certificates software-properties-common
RUN apt-get install -y libharfbuzz-dev libfribidi-dev
RUN apt-get -y install libcurl4-gnutls-dev

# set up the key for adding the R repo
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7

# add the correct Linux R repo
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/debian bullseye-cran40/'

# re-update based on previous setup
RUN apt-get update && apt-get -y upgrade

# install gcc and R
ARG rversion=4.1.2-1
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

RUN apt-get install -y r-cran-data.table r-cran-doparallel r-cran-foreach r-cran-biocmanager r-cran-devtools

# Install arrow from rspm
RUN R -e "options(BioC_mirror = 'https://packagemanager.rstudio.com/all/__linux__/bullseye/latest', HTTPUserAgent = sprintf(\"R/%s R (%s)\", getRversion(), paste(getRversion(), R.version[\"platform\"], R.version[\"arch\"], R.version[\"os\"])))"
RUN R -e "install.packages('arrow', repos = 'https://packagemanager.rstudio.com/all/__linux__/bullseye/latest')"

#install flowsom requirements
RUN apt-get install -y r-cran-igraph r-bioc-biocgenerics r-bioc-consensusclusterplus r-cran-dplyr r-cran-ggforce r-cran-ggplot2 r-cran-ggpubr r-cran-ggrepel r-cran-magrittr r-cran-pheatmap r-cran-rlang r-cran-rtsne r-cran-tidyr r-cran-xml r-cran-scattermore
#install flowsom dependency requirements (eye-roll)
RUN apt-get install -y r-cran-rcppparallel r-bioc-biobase r-cran-matrixstats r-cran-png r-cran-jpeg r-cran-interp r-cran-mass r-bioc-graph r-bioc-rbgl r-cran-scales r-cran-digest r-cran-bh r-cran-rcpparmadillo r-cran-jsonlite r-cran-base64enc r-cran-plyr r-bioc-zlibbioc r-cran-hexbin r-cran-gridextra r-cran-yaml r-bioc-rhdf5lib r-cran-corpcor r-cran-runit r-cran-tibble r-cran-xml2 r-cran-tweenr r-cran-gtable r-cran-polyclip r-cran-tidyselect r-cran-withr r-cran-lifecycle r-cran-rcppeigen

#RUN R -e "library(BiocManager); BiocManager::install('FlowSOM')"
RUN R -e "library(devtools); devtools::install_github('angelolab/FlowSOM', upgrade = FALSE, upgrade_dependencies = FALSE)"

# Stage 3: Adding ark-analysis (Python)
FROM r_deps AS ark_py
COPY --from=r_deps . .
# copy over the requirements.txt, install dependencies, and README
COPY setup.py pyproject.toml requirements.txt README.md start_jupyter.sh /opt/ark-analysis/
RUN python -m pip install -r /opt/ark-analysis/requirements.txt

# copy the scripts over
# this should catch changes to the scripts from updates
COPY ark /opt/ark-analysis/ark

# Install the package via setup.py
RUN cd /opt/ark-analysis && python -m pip install .

# Stage 4: Set the working directory
FROM ark_py as build_output
COPY --from=ark_py . .
WORKDIR /opt/ark-analysis

# jupyter lab
CMD bash start_jupyter.sh