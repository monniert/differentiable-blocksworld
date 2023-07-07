set -e
CUDA_VISIBLE_DEVICES=$cuda python src/ems_eval.py --tag default_raw
CUDA_VISIBLE_DEVICES=$cuda python src/ems_eval.py --tag default_noground
