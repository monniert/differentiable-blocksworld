set -e
CUDA_VISIBLE_DEVICES=$cuda python src/mbf_eval.py --tag default_raw
CUDA_VISIBLE_DEVICES=$cuda python src/mbf_eval.py --tag default_noground
