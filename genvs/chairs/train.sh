#!/bin/sh

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --standalone \
          --nproc_per_node=7 train.py --outdir=training_runs \
                                      --data=/workspace/data/srnchairs \
                                      --cond=0 \
                                      --lr=2e-5 \
                                      --arch=ddpmpp \
                                      --batch=84 \
                                      --duration=200 \
                                      --log_freq=1000 \
                                      --tick=400 \
                                      --snap=5 \
                                      --dump=5 \
                                      --seed=12345 \
                                      --cam_weighting_method=baseline_mean \
