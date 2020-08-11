#!/bin/bash
if [ -d "$PWD/scripts" ]
then
  for f in "$PWD"/templates/*.ipynb
  do
    name=$(basename "$f")
    DIFF=$(diff "$f" "$PWD/scripts/$name" 2>/dev/null)
    DIFFEXIT=$?
    if [ $DIFFEXIT -ne 0 ] && [ $DIFFEXIT -ne 1 ]
    then
      echo "$name was not found.  Creating new file $name in scripts"
      cp -- "$f" "$PWD/scripts/$name"
    elif [ "$DIFF" != "" ]
    then
      echo "Changes have been made to $name; adding updated version as updated_$name"
      echo "Please add relevant changes from updated_$name into $name and remove updated_$name"
      echo "updated_$name will be overwritten in future updates"
      if [ -f "$PWD/scripts/updated_$name" ]
      then
        echo "WARNING: The file updated_$name is being overwritten..."
        rm "$PWD/scripts/updated_$name"
      fi
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
