set -e
CUDA_VISIBLE_DEVICES=$cuda python src/dtu_3d_process.py --name ems --tag default_raw
CUDA_VISIBLE_DEVICES=$cuda python src/dtu_3d_process.py --name ems --tag default_noground --filter_ground
