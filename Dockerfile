FROM python:3.6

# system maintenance
RUN apt-get update && apt-get install -y gcc

WORKDIR /scripts

# copy over the requirements.txt and install dependencies
COPY requirements.txt .
COPY test_python_script.py .

RUN pip install --no-cache-dir -r requirements.txt

# copy the scripts over
COPY segmentation /segmentation
COPY scripts /scripts

# jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
