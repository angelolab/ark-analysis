FROM python:3.6

# system maintenance
RUN apt-get update && apt-get install -y gcc

WORKDIR /scripts

# copy over the requirements.txt and install dependencies
COPY setup.py requirements.txt /opt/segmentation/
RUN pip install -r /opt/segmentation/requirements.txt

# copy the scripts over
COPY segmentation /opt/segmentation/segmentation
COPY scripts /scripts

# Install the package via setup.py
RUN pip install /opt/segmentation

# jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
