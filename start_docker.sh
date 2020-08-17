#!/bin/bash

# behavior depends on scripts directory's presence
if [ -d "$PWD/scripts" ]
then

  # check for update flag
  update=0
  while test $# -gt 0
  do
    case "$1" in
      -u|--update)
        update=1
        shift
        ;;
      *)
        shift
        echo "$1 is not an accepted option..."
        ;;
    esac
  done

  # perform update if requested
  if [ $update -ne 0]
  then
    # check for each template's existance
    for f in "$PWD"/templates/*.ipynb
    do
      # get basename of notebook
      name=$(basename "$f")

      # get difference between similarly named notebook in scripts folder
      DIFF=$(diff "$f" "$PWD/scripts/$name" 2>/dev/null)

      # get exit-code of diff
      #   *-------------------------------------------------------------------*
      #   |  0  |   successful and no differences.  systematically univeral   |
      #   |  1  |   successful and differences.  systematically universal     |
      #   | -1  |   error (no file, etc.).  error code is system dependent    |
      #   |  2  |   error (no file, etc.).  error code is system dependent    |
      #   *-------------------------------------------------------------------*
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
  then
    # -n for no overwrite
    # this is to prevent the case of an empty scripts directory not being filled
    cp -n "$PWD"/templates/*.ipynb "$PWD/scripts/."
  fi
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
