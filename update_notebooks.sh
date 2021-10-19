#!/bin/bash

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
  # if difference, add updated file
  elif [ "$DIFF" != "" ]
  then
    echo "WARNING: The file $name is being overwritten..."
    cp -- "$f" "$PWD/scripts/$name"
  fi
done