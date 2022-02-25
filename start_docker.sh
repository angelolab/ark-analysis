#!/usr/bin/env bash

# check for template developer flag
JUPYTER_DIR='scripts'
update=0
external=''
while test $# -gt 0
do
  case "$1" in
    -d|--develop-notebook-templates)
      JUPYTER_DIR='templates_ark'
      shift
      ;;
    -u|--update)
      update=1
      shift
      ;;
    -e|--external)
      external="$2"
      shift
      shift
      ;;
    *)
      echo "$1 is not an accepted option..."
      echo "-d, --develop-notebook-templates  : Mount templates for direct editing."
      echo "-u, --update                      : Update default scripts"
      echo "-e, --external                    : Mount external drives to /data/external"
      exit
      ;;
  esac
done

# if requirements.txt has been changed in the last half hour, automatically rebuild Docker first
if [[ $(find . -mmin -30 -type f -print | grep requirements.txt | wc -l) -eq 1 ]]
  then
    echo "New requirements.txt file detected, rebuilding Docker"
    docker build -t ark-analysis .
fi

if [ $update -ne 0 ]
  then
    bash update_notebooks.sh -u
  else
    bash update_notebooks.sh
fi

# find lowest open port available
PORT=8888

until [[ $(docker container ls | grep 0.0.0.0:$PORT | wc -l) -eq 0 ]]
  do
    ((PORT=$PORT+1))
done

if [ ! -z "$external" ]
  then
    docker run -it \
      -p $PORT:$PORT \
      -e JUPYTER_PORT=$PORT\
      -v "$PWD/ark:/usr/local/lib/python3.6/site-packages/ark" \
      -v "$PWD/scripts:/scripts" \
      -v "$PWD/data:/data" \
      -v "$external:/data/external" \
      -v "$PWD/ark/phenotyping/create_pixel_som.R:/create_pixel_som.R" \
      -v "$PWD/ark/phenotyping/run_pixel_som.R:/run_pixel_som.R" \
      -v "$PWD/ark/phenotyping/pixel_consensus_cluster.R:/pixel_consensus_cluster.R" \
      -v "$PWD/ark/phenotyping/create_cell_som.R:/create_cell_som.R" \
      -v "$PWD/ark/phenotyping/run_cell_som.R:/run_cell_som.R" \
      -v "$PWD/ark/phenotyping/cell_consensus_cluster.R:/cell_consensus_cluster.R" \
      -v "$PWD/.toks:/home/.toks" \
      angelolab/ark-analysis
  else
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
      -v "$PWD/.toks:/home/.toks" \
      angelolab/ark-analysis
fi
