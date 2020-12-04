FROM python:3.6

# system maintenance
RUN apt-get update && apt-get install -y gcc

WORKDIR /scripts

# copy over the requirements.txt, install dependencies, and README
COPY setup.py requirements.txt requirements-nodeps.txt README.md /opt/ark-analysis/
RUN pip install -r /opt/ark-analysis/requirements.txt

# install mibilib separately, with no dependencies, to avoid irrelevant dependency conflicts
RUN pip install --no-deps -r /opt/ark-analysis/requirements-nodeps.txt

# copy the scripts over
COPY ark /opt/ark-analysis/ark

# Install the package via setup.py
RUN pip install /opt/ark-analysis

# jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
