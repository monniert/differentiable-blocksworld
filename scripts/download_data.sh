#!/bin/bash
set -e
echo "0 - DTU"
echo "1 - BlendedMVS"
echo "2 - Nerfstudio"
read -p "Enter the dataset ID you want to download: " ds_id

mkdir -p datasets
cd datasets

if [ $ds_id == 0 ]
then
    echo "start downloading DTU..."
    wget https://www.dropbox.com/s/bl5j5pfczf90lmr/DTU.zip
    echo "done, start unzipping..."
    unzip DTU.zip
    rm -rf __MACOSX
    echo "done"

elif [ $ds_id == 1 ]
then
    echo "start downloading BlendedMVS..."
    wget https://www.dropbox.com/s/c88216wzn9t6pj8/BlendedMVS.zip
    echo "done, start unzipping..."
    unzip BlendedMVS.zip
    rm -rf __MACOSX
    echo "done"

elif [ $ds_id == 2 ]
then
    echo "start downloading Nerfstudio..."
    gdown --no-cookies 1wsUVqJlsZY-dp9dSemghGe0ijOo9AOM5 -O nerfstudio.zip
    echo "done, start unzipping..."
    unzip nerfstudio.zip
    echo "done, creating data and outputs symlinks..."
    cd ..
    ln -s datasets/nerfstudio/data data
    ln -s datasets/nerfstudio/outputs outputs
    cd datasets
    echo "done"

else
    echo "You entered an invalid ID!"
fi

cd ..
