#!/bin/bash
NV_GPU='0' docker run -it \
  -p 8888:8888 \
  -v $PWD/segmentation:/usr/local/lib/python3.5/dist-packages/segmentation/ \
  -v $PWD/scripts:/scripts \
  -v $PWD/data:/data \
  $USER/segmentation:latest
