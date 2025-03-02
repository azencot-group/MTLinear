#!/bin/bash

epochs=20
multi_es=1 # multi early stopping
model=MTLinear 
use_horizon_penalty=1 
use_variates_penalty=1 
seq=36 # sequence length
itr=3 #number of iterations
lr=0.01 #learning rate
seed=2021

# layer type
lt=DLinear

# grid params
for pp in 1 2
do
for cd in 1 0.5 0.293 0.134
do

for pl in 24 36 48 60
do

seq=36
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --model_id ili \
 --model $model \
 --data custom \
 --features M \
 --seq_len $seq\
 --pred_len $pl \
 --label_len 18 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32

done



for pl in 96 192 336 720
do

seq=96
# weather 
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --model_id weather \
 --model $model \
 --data custom \
 --features M \
 --seq_len $seq\
 --pred_len $pl \
 --label_len 48 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32


# traffic  
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --model_id traffic \
 --model $model \
 --data custom \
 --features M \
 --seq_len $seq\
 --pred_len $pl \
 --label_len 48 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32


 # ECL
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/electricity/ \
 --data_path electricity.csv \
 --model_id ECL \
 --model $model \
 --data custom \
 --features M \
 --seq_len $seq\
 --label_len 48 \
 --pred_len $pl \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32



# exchange
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model_id Exchange \
 --model $model \
 --data custom \
 --features M \
 --seq_len $seq \
 --pred_len $pl \
 --label_len 48 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32


# ETTm1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1 \
  --model $model \
  --data ETTm1 \
  --features M \
  --seq_len $seq\
  --pred_len $pl \
  --label_len 48 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
    --itr $itr \
    --learning_rate $lr\
    --train_epochs $epochs\
    --seed $seed\
    --cluster_dist $cd\
    --layer_type $lt\
    --penalty_param $pp\
    --multi_early_stopping $multi_es\
    --use_horizon_penalty $use_horizon_penalty\
    --use_variates_penalty $use_variates_penalty\
    --batch_size 32


# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2 \
  --model $model \
  --data ETTm2 \
  --features M \
  --seq_len $seq\
  --pred_len $pl \
  --label_len 48 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32



# ETTh1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len $seq\
  --pred_len $pl \
  --label_len 48 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32


  # ETTh2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2 \
  --model $model \
  --data ETTh2 \
  --features M \
  --seq_len $seq\
  --pred_len $pl \
  --label_len 48 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
 --itr $itr \
 --learning_rate $lr\
 --train_epochs $epochs\
 --seed $seed\
 --cluster_dist $cd\
 --layer_type $lt\
 --penalty_param $pp\
 --multi_early_stopping $multi_es\
 --use_horizon_penalty $use_horizon_penalty\
 --use_variates_penalty $use_variates_penalty\
 --batch_size 32


done
done
done