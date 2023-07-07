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
for i in {0..4}
do
    seed=$(shuf -i 1-1000000 -n 1)
    if [ -z "$default" ]
    then
        sed -i "s/seed:.*/seed: $seed/" configs/$config
    else
        sed -i "s/seed:.*/seed: $seed/" configs/$default
    fi

    CUDA_VISIBLE_DEVICES=$cuda python src/trainer.py --tag ${run_tag}_$i --config $config --default $default
done
