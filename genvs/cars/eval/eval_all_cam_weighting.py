# written by JhihYang Wu <jhihyangwu@arizona.edu>
# script for evaluating the all the camera weighting algorithms on GeNVS
# to run simply execute: python eval_all_cam_weighting.py GPU_ID
# make sure to replace GPU_ID with which GPU to run on

import os
import sys

# what to evaluate
EVAL_METHODS = [  # see genvs/models/cam_weighting.py for implementation
    "baseline_mean",
    "distance",
    "error_weighing_alpha=0.5",
    "error_weighing_alpha=0.25",
    "error_weighing_alpha=0.2",
    "error_weighing_alpha=0.3",
    "distance_gaussian_b=3",
    "distance_gaussian_b=1",
    "distance_gaussian_b=0.3",
    "distance_gaussian_b=0.1",
    "l1_weighing",
    "f_norm_weighting",
    "rel_cam_poses_l2",
    "rel_cam_poses_f_norm",
    "distance_squared",
    "distance_cubed",
    "error_weighing_alpha=0.6",
    "error_weighing_alpha=0.75",
    "error_weighing_alpha=0.8",
    "error_weighing_alpha=0.9",
]

# constants
OUT_DIR = "cam_weighting_eval_out/"

def main():
    gpu_id = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    for method in EVAL_METHODS:
        dir_name = f"input_views=56_58_67_75_91_113_241_244_sum_target_views=1045_method={method}"
        path = os.path.join(OUT_DIR, dir_name)
        if os.path.exists(path):
            print(f"skipping {path}")
        else:
            print(f"evaluating {path}")
            eval(method, path, gpu_id)

def eval(method, out_path, gpu_id):
    os.system(f"""
        python eval.py \
            --out_dir={out_path} \
            --denoising_steps=25 \
            --cam_weighting='{method}' \
            --ckpt_path=/workspace/data/ourgenvs_trained_1/network-snapshot-004000.pkl \
            --gpu_id={gpu_id} \
            --data_path=/workspace/data/srncars/cars \
            --input_views='56 58 67 75 91 113 241 244' \
            --target_views='40 44 54 149 159 173 181 245' \
            --batch_size=32    
    """)

if __name__ == "__main__":
    main()
