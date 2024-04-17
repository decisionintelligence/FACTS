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

#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 15 --range 350 525 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 16 --range 1020 1210 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 17 --range 1833 1963 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 18 --range 2435 2618 &
#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 19 --range 525 700 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 20 --range 1210 1400 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 21 --range 1963 2100 &
#CUDA_VISIBLE_DEVICES=7 python ../exps/collect_seeds.py --gpu_id 22 --range 2618 2800 &

#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 23 --range 9864 9932 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 24 --range 9932 10000 &

CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 35 --range 4169 4184 &
CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 36 --range 4184 4200 &


#3090 7
#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 9 --range 5600 6350 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 10 --range 6350 7100 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 11 --range 7100 7850 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 12 --range 7850 8600 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 13 --range 8600 9300 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 14 --range 9300 10000 &


#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 26 --range 6934 7100 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 27 --range 7610 7850 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 28 --range 8407 8600 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 29 --range 9067 9300 &

CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 25 --range 6135 6350 &
CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 30 --range 7087 7100 &
CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 31 --range 7720 7850 &
CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 32 --range 8519 8600 &
CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 33 --range 9167 9300 &

CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 34 --range 9182 9300 &
