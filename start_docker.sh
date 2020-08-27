#!/bin/bash

# catch and pass update flag to notebook update routine
bash update_notebooks.sh "$@"

# run docker and start notebook server
docker run -it \
  -p 8888:8888 \
  -v "$PWD/ark:/usr/local/lib/python3.6/site-packages/ark" \
  -v "$PWD/scripts:/scripts" \
  -v "$PWD/data:/data" \
  ark-analysis:latest
