#!/bin/bash
docker run -it \
  -p 8888:8888 \
  -v $PWD/scripts:/scripts \
  -v $PWD/data:/data \
  $USER/segmentation:latest
