#!/usr/bin/env bash

# define the version number, this needs to be updated every new Docker release
VERSION='v0.6.3'

# check for template developer flag
JUPYTER_DIR='scripts'
update=0
external=''
while test $# -gt 0
do
  case "$1" in
    -n|--develop-notebook-templates)
      JUPYTER_DIR='templates'
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

# update the notebooks in the scripts folder if flag set
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

# define the run parameters
run_params=(
  -p $PORT:$PORT
  -e JUPYTER_PORT=$PORT
  -e JUPYTER_DIR=$JUPYTER_DIR
  -e UPDATE_ARK=$update
  -v "$PWD/README.md:/opt/ark-analysis/README.md"
  -v "$PWD/setup.py:/opt/ark-analysis/setup.py"
  -v "$PWD/pyproject.toml:/opt/ark-analysis/pyproject.toml"
  -v "$PWD/start_jupyter.sh:/opt/ark-analysis/start_jupyter.sh"
  -v "$PWD/src:/opt/ark-analysis/src"
  -v "$PWD/scripts:/scripts"
  -v "$PWD/data:/data"
  -v "$PWD/.git:/opt/ark-analysis/.git"
)
[[ ! -z "$external" ]] && run_params+=(-v "$external:/data/external")

# remove the old Docker container if one exists, as it may contain different external volumes
docker rm -f $VERSION > /dev/null 2>&1 || true

# create the Docker container
docker run -it "${run_params[@]}" --name $VERSION angelolab/ark-analysis:$VERSION
