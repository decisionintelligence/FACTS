#!/bin/bash
export PYTHONPATH=../

#A800 8
#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 1 --range 0 650 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 2 --range 650 1300 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 3 --range 1300 1950 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 4 --range 1950 2600 &
#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 5 --range 2600 3250 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 6 --range 3250 3900 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 7 --range 3900 4550 &
#CUDA_VISIBLE_DEVICES=7 python ../exps/collect_seeds.py --gpu_id 8 --range 4550 5200 &

CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 16 --range 2456 2530 &
CUDA_VISIBLE_DEVICES=7 python ../exps/collect_seeds.py --gpu_id 17 --range 2530 2600 &


#3090 7
CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 9 --range 5200 5900 &
CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 10 --range 5900 6600 &
CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 11 --range 6600 7300 &
CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 12 --range 7300 8000 &
CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 13 --range 8000 8700 &
CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 14 --range 8700 9400 &
CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 15 --range 9400 10000 &

CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 18 --range 7170 7235 &
CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 19 --range 7235 7300 &
