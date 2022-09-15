if [ $UPDATE_ARK -ne 0 ]
  then
    cd /opt/ark-analysis && python -m pip install .
fi

cd /scripts
jupyter lab --ip=0.0.0.0 --allow-root --no-browser --port=$JUPYTER_PORT --notebook-dir=/$JUPYTER_DIR