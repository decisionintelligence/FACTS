#!/bin/bash
export PYTHONPATH=../
#A800 8
#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 1 --range 0 350 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 2 --range 350 700 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 3 --range 700 1100 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 4 --range 1100 1500 &
#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 5 --range 1500 2300 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 6 --range 2300 3100 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 7 --range 3100 3900 &
#CUDA_VISIBLE_DEVICES=7 python ../exps/collect_seeds.py --gpu_id 8 --range 3900 4700 &

CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 30 --range 2861 2980 &
CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 31 --range 2980 3100 &


CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 32 --range 4456 4580 &
CUDA_VISIBLE_DEVICES=7 python ../exps/collect_seeds.py --gpu_id 33 --range 4580 4700 &

#3090 7
#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 9 --range 4700 5500 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 10 --range 5500 6300 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 11 --range 6300 7100 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 12 --range 7100 7900 &
#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 13 --range 7900 8700 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 14 --range 8700 8804 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 15 --range 9500 10000 &
#
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 16 --range 8804 8820 &

CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 17 --range 4925 5500 &
CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 18 --range 5668 6257 &
CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 19 --range 6475 6626 &
CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 20 --range 7320 7357 &
CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 21 --range 8055 8226 &
CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 22 --range 8820 8826 &
CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 23 --range 9675 9826 &

CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 24 --range 8826 8857 &

CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 25 --range 7357 7900 &

CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 26 --range 6626 7100 &

CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 27 --range 8857 9500 &

CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 28 --range 8226 8700 &

CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 29 --range 9826 10000 &

CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 34 --range 6796 6950 &
CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 35 --range 6950 7100 &




CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 36 --range 6257 6300 &
CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 37 --range 9349 9424 &
CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 38 --range 9424 9500 &

