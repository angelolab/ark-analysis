#!/bin/bash
docker run -it \
  -p 8888:8888 \
  -v $PWD/segmentation:/usr/local/lib/python3.6/site-packages/segmentation \
  -v $PWD/scripts:/scripts \
  -v $PWD/data:/data \
  segmentation:latest
