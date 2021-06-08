#!/bin/bash

# catch and pass update flag to notebook update routine
bash update_notebooks.sh "$@"

# find lowest open port available
PORT=8888

until [[ $(lsof -i -P -n | grep 127.0.0.1:$PORT | wc -l) -eq 0 ]]
  do
    ((PORT=PORT+1))
done

if [ $PORT -ne 8888 ]
then
  echo "WARNING: another Jupyter server on port 8888 running"
  echo "Enter this URL instead to access the notebooks: http://127.0.0.1:$PORT/"
  echo "In the URLs below, copy the token after \"token=\", paste that into the password prompt, and log in"
fi

docker run -it \
  -p $PORT:8888 \
  -v "$PWD/ark:/usr/local/lib/python3.6/site-packages/ark" \
  -v "$PWD/scripts:/scripts" \
  -v "$PWD/data:/data" \
  -v "$PWD/.toks:/home/.toks"
  ark-analysis:latest
