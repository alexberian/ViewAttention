# written by JhihYang Wu <jhihyangwu@arizona.edu>
# eval_all_cam_weighting.py gets psnr and ssim of all weighting algos
# after that script runs completely, run this script to get fid lpips and dists scores
# to run simply execute: python eval_all_cam_weighting_other_metrics.py

from eval_all_cam_weighting import OUT_DIR
import os

def main():
    for folder in os.listdir(OUT_DIR):
        if not folder.endswith("tmp_fid"):
            print(folder)
            os.system(f"python eval_fid_lpips_dists.py {os.path.join(OUT_DIR, folder)}")

if __name__ == "__main__":
    main()
