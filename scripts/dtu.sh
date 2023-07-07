set -e
for id in 24 31 40 45 55 59 63 75 83 105
do
    cuda=$cuda tag=${tag}_scan${id} config=dtu/scans/scan${id}.yml ./scripts/pipeline.sh
done
