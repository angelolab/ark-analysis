#!/bin/bash

# catch and pass update flag to notebook update routine
bash update_notebooks.sh "$@"

# find lowest open port available
PORT=8888

until [[ $(lsof -i -P -n | grep 127.0.0.1:$PORT | wc -l) -eq 0 ]]
  do
    ((PORT=PORT+1))
done

echo $PORT


# run docker and start notebook server
docker run -it \
  -p $PORT:8888 \
  -v "$PWD/ark:/usr/local/lib/python3.6/site-packages/ark" \
  -v "$PWD/scripts:/scripts" \
  -v "$PWD/data:/data" \
  ark-analysis:latest
