#!/bin/bash

# behavior depends on scripts directory's presence
if [ -d "$PWD/scripts" ]
then
  # check for each template's existance
  for f in "$PWD"/templates/*.ipynb
  do
    # get basename of notebook
    name=$(basename "$f")

    # get difference between similarly named notebook in scripts folder
    DIFF=$(diff "$f" "$PWD/scripts/$name" 2>/dev/null)

    # get exitcode of diff
    #
    # 0       :  successful and no differences
    # 1       :  successful and differences
    # -1 or 2 :  error (no file, etc.)  Exact exit code is system dependent
    #
    DIFFEXIT=$?

    # check for error
    if [ $DIFFEXIT -ne 0 ] && [ $DIFFEXIT -ne 1 ]
    then
      echo "$name was not found.  Creating new file $name in scripts"
      cp -- "$f" "$PWD/scripts/$name"
    
    # if no difference, add updated file
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
  # since there is no scripts directory, just make one and copy from templates
  mkdir "$PWD/scripts"
  cp "$PWD"/templates/*.ipynb "$PWD/scripts/."
fi

# run docker and start notebook server
docker run -it \
  -p 8888:8888 \
  -v "$PWD/ark:/usr/local/lib/python3.6/site-packages/ark" \
  -v "$PWD/scripts:/scripts" \
  -v "$PWD/data:/data" \
  ark-analysis:latest
