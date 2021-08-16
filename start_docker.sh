#!/usr/bin/env bash

# if requirements.txt has been changed in the last day, automatically rebuild Docker first
if [[ $(find . -mmin -1440 -type f -print | grep requirements.txt | wc -l) -eq 1 ]]
  then
    echo "New requirements.txt file detected, rebuilding Docker"
    docker build -t ark-analysis .
fi

# catch and pass update flag to notebook update routine
bash update_notebooks.sh "$@"

# find lowest open port available
PORT=8888

until [[ $(docker container ls | grep 0.0.0.0:$PORT | wc -l) -eq 0 ]]
  do
    ((PORT=$PORT+1))
done

docker run -it \
  -p $PORT:$PORT \
  -e JUPYTER_PORT=$PORT\
  -v "$PWD/ark:/usr/local/lib/python3.6/site-packages/ark" \
  -v "$PWD/scripts:/scripts" \
  -v "$PWD/data:/data" \
  -v "$PWD/ark/phenotyping/create_pixel_som.R:/create_pixel_som.R" \
  -v "$PWD/ark/phenotyping/run_pixel_som.R:/run_pixel_som.R" \
  -v "$PWD/ark/phenotyping/pixel_consensus_cluster.R:/pixel_consensus_cluster.R" \
  -v "$PWD/ark/phenotyping/create_cell_som.R:/create_cell_som.R" \
  -v "$PWD/ark/phenotyping/run_cell_som.R:/run_cell_som.R" \
  -v "$PWD/ark/phenotyping/cell_consensus_cluster.R:/cell_consensus_cluster.R" \
  -v "$PWD/.toks:/home/.toks"
  ark-analysis:latest
