#!/bin/bash
if [ -d "$PWD/scripts" ]
then
  for f in "$PWD"/templates/*.ipynb
  do
    name=$(basename "$f")
    DIFF=$(diff "$f" "$PWD/scripts/$name")
    if [ "$DIFF" != "" ]
    then
      echo "Changes have been made to $name; adding updated version as updated_$name"
      cp -- "$f" "$PWD/scripts/updated_$name"
    fi
  done
else
  mkdir "$PWD/scripts"
  cp "$PWD"/templates/*.ipynb "$PWD/scripts/."
fi

docker run -it \
  -p 8888:8888 \
  -v "$PWD/segmentation:/usr/local/lib/python3.6/site-packages/segmentation" \
  -v "$PWD/scripts:/scripts" \
  -v "$PWD/data:/data" \
  segmentation:latest
