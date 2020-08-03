#!/bin/bash
if [ -d "$PWD/scripts" ]
then
  for f in "$PWD"/.templates/*.ipynb
  do
    name=$(basename "$f")
    cp -- "$f" "$PWD/scripts/updated_$name"
  done
else
  mkdir "$PWD/scripts"
  cp "$PWD"/.templates/*.ipynb "$PWD/scripts/."
fi

docker run -it \
  -p 8888:8888 \
  -v "$PWD/segmentation:/usr/local/lib/python3.6/site-packages/segmentation" \
  -v "$PWD/scripts:/scripts" \
  -v "$PWD/data:/data" \
  segmentation:latest
