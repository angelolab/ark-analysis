#!/usr/bin/env bash

# check for template developer flag
JUPYTER_DIR='scripts'
update=0
external=''
while test $# -gt 0
do
  case "$1" in
    -n|--develop-notebook-templates)
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
      echo "-n, --develop-notebook-templates  : Mount templates for direct editing."
      echo "-u, --update                      : Update default scripts and code changes"
      echo "-e, --external                    : Mount external drives to /data/external"
      exit
      ;;
  esac
done

# find lowest open port available
PORT=8888

if [ $update -ne 0 ]
  then
    bash update_notebooks.sh -u
  else
    bash update_notebooks.sh
fi

until [[ $(docker container ls | grep 0.0.0.0:$PORT | wc -l) -eq 0 ]]
  do
    ((PORT=$PORT+1))
done

run_params=(
  -p $PORT:$PORT
  -e JUPYTER_PORT=$PORT
  -e JUPYTER_DIR=$JUPYTER_DIR
  -e UPDATE_ARK=$update
  -v "$PWD/README.md:/opt/ark-analysis/README.md"
  -v "$PWD/setup.py:/opt/ark-analysis/setup.py"
  -v "$PWD/requirements.txt:/opt/ark-analysis/requirements.txt"
  -v "$PWD/pyproject.toml:/opt/ark-analysis/pyproject.toml"
  -v "$PWD/start_jupyter.sh:/opt/ark-analysis/start_jupyter.sh"
  -v "$PWD/ark:/opt/ark-analysis/ark"
  -v "$PWD/scripts:/scripts"
  -v "$PWD/data:/data"
  -v "$PWD/ark/phenotyping/create_pixel_som.R:/create_pixel_som.R"
  -v "$PWD/ark/phenotyping/run_pixel_som.R:/run_pixel_som.R"
  -v "$PWD/ark/phenotyping/pixel_consensus_cluster.R:/pixel_consensus_cluster.R"
  -v "$PWD/ark/phenotyping/create_cell_som.R:/create_cell_som.R"
  -v "$PWD/ark/phenotyping/run_cell_som.R:/run_cell_som.R"
  -v "$PWD/ark/phenotyping/cell_consensus_cluster.R:/cell_consensus_cluster.R"
  -v "$PWD/.toks:/home/.toks"
)
[[ ! -z "$external" ]] && run_params+=(-v "$external:/data/external")

docker run -it "${run_params[@]}" angelolab/ark-analysis:v0.4.2
