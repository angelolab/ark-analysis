#!/usr/bin/env bash

# check for template developer flag
JUPYTER_DIR='scripts'
update=0

while test $# -gt 0
do
  case "$1" in
    -u|--update)
      update=1
      shift
      ;;
    *)
      echo "$1 is not an accepted option..."
      echo "-u, --update                      : Update default scripts"
      exit
      ;;
  esac
done

if [ $update -ne 0 ]
  then
    bash update_notebooks.sh -u
  else
    bash update_notebooks.sh
fi

jupyter lab --notebook-dir $JUPYTER_DIR
