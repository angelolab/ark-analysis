FROM python:3.6

# system maintenance
RUN apt-get update && apt-get install -y gcc

WORKDIR /scripts

# copy over the requirements.txt, install dependencies, and README
COPY setup.py pyproject.toml requirements.txt README.md /opt/ark-analysis/
COPY .toks /home/.toks
RUN python -m pip install -r /opt/ark-analysis/requirements.txt

# copy the scripts over
COPY ark /opt/ark-analysis/ark

# Install the package via setup.py
RUN cd /opt/ark-analysis && python -m pip install -e .
RUN cd /opt/ark-analysis && python -m pip install .

# jupyter lab
CMD jupyter lab --ip=0.0.0.0 --allow-root --no-browser --port=$JUPYTER_PORT
