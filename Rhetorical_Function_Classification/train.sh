#!/bin/bash

export PYTHONPATH=.

python train.py  --mode train --cuda  --rhe_data_dir ./rhetorical_data  --data_dir ./rhetorical_data --batch_size 2 --seed 666 --train_steps 3000 --save_checkpoint_steps 100  --report_every 1  --visible_gpus 0 --gpu_ranks 0  --world_size 1 --accum_count 1 --dec_dropout 0.1 --enc_dropout 0.1  --model_path  ./trained_model/train_sentrhe  --inter_layers 6,7  --inter_heads 8 --hier --doc_max_timesteps 50 --num_workers 5 --warmup_steps 8000 --lr 0.00002 --enc_layers 6  --dec_layers 6 
