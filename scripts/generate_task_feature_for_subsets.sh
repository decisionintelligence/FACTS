#!/bin/bash

export PYTHONPATH=../
export CUDA_VISIBLE_DEVICES=4

date=1123
repr_dims=256
sample_num=100
epochs=20
loader='subset'
train_script="../exps/generate_task_feature.py"



seq_len=12
base="pems/PEMS03"
dataset=("7" "19" "24" "28" "37" "64" "0" "4" "10" "18" "20" "27")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS03"
dataset=("1" "3" "17" "21" "23" "62" "2" "5" "12" "14" "22" "40")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="pems/PEMS04"
dataset=("11" "14" "25" "32" "44" "53" "0" "1" "5" "13" "17" "50")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS04"
dataset=("10" "16" "20" "25" "33" "42" "4" "6" "15" "18" "29" "30")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="pems/PEMS07"
dataset=("0" "8" "15" "25" "44" "66" "1" "3" "4" "10" "17" "32")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS07"
dataset=("5" "13" "35" "21" "28" "43" "2" "7" "14" "19" "60" "66")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="pems/PEMS08"
dataset=("5" "19" "25" "46" "61" "70" "3" "9" "10" "20" "29" "34")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="pems/PEMS08"
dataset=("4" "10" "25" "30" "41" "64" "11" "3" "12" "17" "31" "40")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=12
base="METR-LA/metr-la"
dataset=("1" "8" "26" "30" "45" "62" "3" "7" "20" "25" "35" "50")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="METR-LA/metr-la"
dataset=("5" "9" "2" "34" "41" "63" "6" "21" "29" "40" "60" "10")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="solar/solar_AL"
dataset=("1" "8" "25" "47" "62" "73" "2" "14" "26" "31" "34" "71")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="solar/solar_AL"
dataset=("4" "13" "21" "46" "47" "61" "5" "7" "12" "16" "20" "59")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=12
base="ETT-small/ETTh1"
dataset=("3" "6" "9" "0" "1" "14")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTh1"
dataset=("2" "5" "13" "7" "8" "12")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="ETT-small/ETTh2"
dataset=("2" "7" "15" "1" "5" "6")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTh2"
dataset=("3" "8" "13" "0" "4" "9")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="ETT-small/ETTm1"
dataset=("2" "5" "14" "0" "1" "13")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTm1"
dataset=("3" "7" "15" "12" "11" "8")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="ETT-small/ETTm2"
dataset=("1" "11" "13" "10" "14" "2")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="ETT-small/ETTm2"
dataset=("0" "8" "15" "7" "6" "9")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done


seq_len=12
base="exchange_rate/exchange_rate"
dataset=("7" "10" "3" "6")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done

seq_len=48
base="exchange_rate/exchange_rate"
dataset=("4" "11" "5" "8")

for set in "${dataset[@]}"; do
    python  $train_script --dataset "${base}_${set}" \
                          $date\
                          --loader $loader \
                          --repr_dims $repr_dims\
                          --seq_len $seq_len \
                          --epochs $epochs \
                          --sample_num $sample_num
done