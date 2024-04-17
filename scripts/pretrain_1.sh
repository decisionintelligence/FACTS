#!/bin/bash
export PYTHONPATH=../

#A800 8
CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 1 --range 0 385 &
CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 2 --range 385 770 &
CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 3 --range 770 1155 &
CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 4 --range 1155 1540 &
CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 5 --range 1540 1925 &
CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 6 --range 1925 2310 &
CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 7 --range 2310 2695 &
CUDA_VISIBLE_DEVICES=7 python ../exps/collect_seeds.py --gpu_id 8 --range 2695 3080 &


#3090 7
#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 9 --range 5800 6400 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 10 --range 6400 7000 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 11 --range 7000 7600 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 12--range 7600 8200 &
#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 13 --range 8200 8800 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 14 --range 8800 9400 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 15 --range 9400 10000 &

#CUDA_VISIBLE_DEVICES=0 python ../exps/collect_seeds.py --gpu_id 16 --range 3080 3465 &
#CUDA_VISIBLE_DEVICES=1 python ../exps/collect_seeds.py --gpu_id 17 --range 3465 3850 &
#CUDA_VISIBLE_DEVICES=2 python ../exps/collect_seeds.py --gpu_id 18 --range 3850 4235 &
#CUDA_VISIBLE_DEVICES=3 python ../exps/collect_seeds.py --gpu_id 19 --range 4235 4620 &
#CUDA_VISIBLE_DEVICES=4 python ../exps/collect_seeds.py --gpu_id 20 --range 4620 5005 &
#CUDA_VISIBLE_DEVICES=5 python ../exps/collect_seeds.py --gpu_id 21 --range 5005 5390 &
#CUDA_VISIBLE_DEVICES=6 python ../exps/collect_seeds.py --gpu_id 22 --range 5390 5800 &
