#!/bin/bash
set -e
if ! [[ ${tag:0:1} =~ ^[0-9] ]]
    then 
        CURRENTDATE=`date +"%m%d"`
        run_tag="${CURRENTDATE}_$tag"
else
    run_tag=$tag
fi
dataset="$(sed -n 's/.*name: //p' configs/$config | sed -n 1p)"
CUDA_VISIBLE_DEVICES=$cuda python src/trainer.py --tag $run_tag --config $config --default $default
