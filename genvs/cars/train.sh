#!/bin/sh

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone \
          --nproc_per_node=4 train.py --outdir=training_runs \
                                      --data=/workspace/data/srncars \
                                      --cond=0 \
                                      --lr=3e-4 \
                                      --arch=ddpmpp \
                                      --batch=48 \
                                      --duration=200 \
                                      --log_freq=1000 \
                                      --tick=400 \
                                      --snap=5 \
                                      --dump=5 \
                                      --seed=12345 \
                                      --cam_weighting_method=baseline_mean \
                                      --transfer=/workspace/data/ourgenvs_trained_1/srncars/network-snapshot-004000.pkl \
