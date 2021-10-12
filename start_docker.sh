#!/usr/bin/env bash

# check for template developer flag
JUPYTER_DIR='scripts'
while test $# -gt 0
do
  case "$1" in
    -d|--develop-notebook-templates)
      JUPYTER_DIR='templates'
      shift
      ;;
    *)
      echo "$1 is not an accepted option..."
      echo "-d, --develop-notebook-templates : Mount templates for direct editing."
      exit
      ;;
  esac
done

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
  -e JUPYTER_DIR=$JUPYTER_DIR\
  -v "$PWD/ark:/usr/local/lib/python3.6/site-packages/ark" \
  -v "$PWD/$JUPYTER_DIR:/$JUPYTER_DIR" \
  -v "$PWD/data:/data" \
  -v "$PWD/.toks:/home/.toks" \
  ark-analysis:latest
