#!/bin/bash
export PYTHONPATH=../
#A800 8
#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 1 --range 0 700 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 2 --range 700 1400 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 3 --range 1400 2100 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 4 --range 2100 2800 &
#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 5 --range 2800 3500 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 6 --range 3500 4200 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 7 --range 4200 4900 &
#CUDA_VISIBLE_DEVICES=7 python ../exps/collect_seeds.py --gpu_id 8 --range 4900 5600 &

CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 15 --range 8079 8340 &
CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 16 --range 8340 8600 &


#3090 6
CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 9 --range 5600 6350 &
CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 10 --range 6350 7100 &
CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 11 --range 7100 7850 &
CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 12 --range 7850 8600 &
CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 13 --range 8600 9300 &
CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 14 --range 9300 10000 &




