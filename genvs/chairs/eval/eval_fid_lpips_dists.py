# written by JhihYang Wu <jhihyangwu@arizona.edu>
# get FID, LPIPS, and DISTS scores of a eval out folder
# usage: python eval_fid_lpips_dists.py path_to_folder_with_finish.txt
# does nothing if already ran before

import sys
import os
from prep_eval_fid import GPU_ID, get_scene_names, GT_DATA_PATH
from tqdm import tqdm
import skimage.metrics
import numpy as np
import torch
import imageio
import subprocess

device = torch.device(f"cuda:{GPU_ID}")
print(f"Using device: {device}")

def main():
    path = get_eval_out_path()   
    scene_names = get_scene_names(path)
    # compute lpips and dists scores
    if not os.path.exists(os.path.join(path, "finish_2.txt")):
        finish_file = open(os.path.join(path, "finish_2.txt"), "w", buffering=1)
        # define metrics to test
        metrics = [calc_psnr, calc_ssim, calc_lpips, calc_dists]
        metric_names = ["psnr", "ssim", "lpips", "dists"]
        metric_totals = [[], [], [], []]
        # start testing
        for scene_name in tqdm(scene_names):
            finish_file.write(scene_name)
            imgs = get_images(path, scene_name)
            for metric_i, metric in enumerate(metrics):
                all_vals = metric(imgs)
                metric_val = all_vals.mean()
                metric_totals[metric_i].append(metric_val)
                finish_file.write(" " + str(metric_val))
            finish_file.write("\n")
        # write final avg of metrics
        metric_avgs = [sum(total) / len(total) for total in metric_totals]
        final_line = [f"{metric_names[i]}: {metric_avgs[i]}" for i in range(len(metric_names))]
        finish_file.write("final " + " | ".join(final_line) + "\n")
        finish_file.close()
    # compute fid score
    if not os.path.exists(os.path.join(path, "finish_3.txt")):
       res1 = subprocess.getoutput(f"python prep_eval_fid.py {path}")
       res1 = res1.split("\n")
       assert res1[-1].startswith("python -m pytorch_fid ")
       res2 = subprocess.getoutput(res1[-1])
       res2 = res2.split("\n")[-1]
       with open(os.path.join(path, "finish_3.txt"), "w") as file:
           file.write(res2)

def get_eval_out_path():
    if len(sys.argv) != 2:
        print("Usage: python eval_fid_lpips_dists.py path_to_folder_with_finish.txt")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(os.path.join(path, "finish.txt")):
        print(f"Bad path {path}")
        sys.exit(1)
    return path if path[-1] != "/" else path[:-1]

def get_images(path, scene_name):
    all_pred_images = []
    all_gt_images = []
    for img_filename in sorted(os.listdir(os.path.join(path, scene_name))):
        pred = imageio.imread(os.path.join(path, scene_name, img_filename))[..., :3]
        all_pred_images.append(pred)
        gt = imageio.imread(os.path.join(GT_DATA_PATH, scene_name, img_filename))[..., :3]
        all_gt_images.append(gt)
    all_pred_images = np.stack(all_pred_images, axis=0)
    all_gt_images = np.stack(all_gt_images, axis=0)
    return all_pred_images, all_gt_images

# === EVALUATION METRICS ===

def calc_psnr(imgs):
    pred, ground = imgs
    # imgs = (pred, ground)
    # pred.shape = (num_imgs, H, W, 3)
    # ground.shape = (num_imgs, H, W, 3)
    retval = []
    for i in range(pred.shape[0]):
        retval.append(skimage.metrics.peak_signal_noise_ratio(
                    pred[i],
                    ground[i],
                    data_range=255
                ))
    return np.array(retval)

def calc_ssim(imgs):
    pred, ground = imgs
    # imgs = (pred, ground)
    # pred.shape = (num_imgs, H, W, 3)
    # ground.shape = (num_imgs, H, W, 3)
    retval = []
    for i in range(pred.shape[0]):
        retval.append(skimage.metrics.structural_similarity(
                    pred[i],
                    ground[i],
                    multichannel=True,
                    data_range=255,
                ))
    return np.array(retval)

import lpips
lpips_alex = lpips.LPIPS(net="alex").to(device=device)
def calc_lpips(imgs):
    # https://github.com/richzhang/PerceptualSimilarity
    pred, ground = imgs
    # imgs = (pred, ground)
    # pred.shape = (num_imgs, H, W, 3)
    # ground.shape = (num_imgs, H, W, 3)

    # cvt to tensor
    pred = torch.from_numpy(pred).to(device)
    ground = torch.from_numpy(ground).to(device)
    # permute
    pred = torch.permute(pred, (0, 3, 1, 2))
    ground = torch.permute(ground, (0, 3, 1, 2))
    # normalize to -1 to +1
    pred = pred / 127.5 - 1.0
    ground = ground / 127.5 - 1.0
    # calc
    retval = lpips_alex(pred, ground)
    retval = retval.cpu().detach().numpy().reshape(-1)
    return retval

from DISTS_pytorch import DISTS
dists_D = DISTS().to(device)
def calc_dists(imgs):
    # https://github.com/dingkeyan93/DISTS
    pred, ground = imgs
    # imgs = (pred, ground)
    # pred.shape = (num_imgs, H, W, 3)
    # ground.shape = (num_imgs, H, W, 3)

    # cvt to tensor
    pred = torch.from_numpy(pred).to(device)
    ground = torch.from_numpy(ground).to(device)
    # permute
    pred = torch.permute(pred, (0, 3, 1, 2))  # (num_imgs, 3, H, W)
    ground = torch.permute(ground, (0, 3, 1, 2))  # (num_imgs, 3, H, W)
    # normalize to 0 to +1
    pred = pred / 255.0
    ground = ground / 255.0
    # calc
    retval = dists_D(pred, ground)
    retval = retval.cpu().detach().numpy().reshape(-1)
    return retval

# ==========================

if __name__ == "__main__":
    main()
