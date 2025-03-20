#!/bin/bash

mkdir -p data
cd data

echo "Downloading the DTU dataset ..."
wget https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip || { echo "wget failed! Trying curl..."; curl -L -o DTU.zip https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip; }

echo "Start unzipping ..."
unzip DTU.zip

echo "DTU dataset is ready!"
