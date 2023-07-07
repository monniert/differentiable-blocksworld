#!/bin/bash
set -e
for config in configs/dtu/*yml
do
    echo $config
    yq eval -i '.model.loss.countsqmean_weight = 0.01' $config
    yq eval -i 'del(.model.loss.countsq_weight)' $config
done
