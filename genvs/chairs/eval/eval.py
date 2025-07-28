# written by JhihYang Wu <jhihyangwu@arizona.edu>
# script for evaluating the performance of GeNVS
# inspired by https://github.com/sxyu/pixel-nerf/blob/master/eval/eval.py

import click
import os
import sys
import pickle
import torch
import skimage.metrics
from tqdm import tqdm
import imageio
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from genvs.data.srnchairs import SRNMultiChairs
from genvs.utils.utils import StackedRandomGenerator, edm_sampler, one_step_edm_sampler, img_tensor_to_npy
from genvs.models.genvs import GeNVS

@click.command()
@click.option("--out_dir", help="Where to save results", required=True)
@click.option("--denoising_steps", help="How many denoising steps to use for diffusion model", required=True)
@click.option("--cam_weighting", help="Which camera weighting algorithm to use", required=True)
@click.option("--ckpt_path", help="Path to trained model", required=True)
@click.option("--gpu_id", help="Which GPU to use", required=True)
@click.option("--data_path", help="Path to dataset", required=True)
@click.option("--input_views", help="Which views to use as input", required=True)
@click.option("--target_views", help="Which views to use as target", required=True)
@click.option("--batch_size", help="How many target views to generate at once", required=True)

def main(**kwargs):
    # make out directory
    os.makedirs(kwargs["out_dir"], exist_ok=False)
    finish_file = open(os.path.join(kwargs["out_dir"], "finish.txt"), "a", buffering=1)

    # setup CUDA device
    device = torch.device(f"cuda:{int(kwargs['gpu_id'])}")
    print(f"Using device: {device}")

    # get dataset
    dataset = SRNMultiChairs(kwargs["data_path"], distr="test")
    _, _, H, W = dataset[0]["images"].shape

    # load trained GeNVS
    net = GeNVS(None, cam_weighting_method=kwargs["cam_weighting"])
    with open(kwargs["ckpt_path"], "rb") as f:
        tmp_net = pickle.load(f)["ema"].cpu()
        net.denoiser = tmp_net.denoiser
        net.load_state_dict(tmp_net.state_dict())
        del tmp_net
    net = net.to(device)
    net.eval()

    # prepare a constant latent to be used by denoiser
    rnd = StackedRandomGenerator(device, [0])
    latents = rnd.randn([1, 3, H, W], device=device)

    # create input and target view indexing tensors
    input_views = list(map(int, kwargs["input_views"].split()))
    input_views = torch.tensor(input_views)
    all_target_views = list(map(int, kwargs["target_views"].split()))
    all_target_views = torch.tensor(all_target_views)

    # evaluate on one scene at a time
    total_psnr = 0.0
    total_ssim = 0.0
    cnt = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            # load data
            data = dataset[i]
            images = data["images"]
            poses = data["poses"].to(device)
            focal = data["focal"].item()
            z_near = data["z_near"]
            z_far = data["z_far"]
            obj_name = os.path.basename(data["path"])
            print(f"OBJECT {i} OF {len(dataset)} {data['path']}")
            os.makedirs(os.path.join(kwargs["out_dir"], obj_name), exist_ok=True)

            # encode input views
            inputs_encoded = net.pixel_nerf_net.encode_input_views(images[input_views][None].to(device),
                                                                poses[input_views][None])

            # generate batch_size target views at a time
            ssims, psnrs = [], []
            for target_views in tqdm(torch.split(all_target_views, int(kwargs["batch_size"]))):
                # 1. render feature images at target_views
                feature_images = net.pixel_nerf_net(inputs_encoded, poses[target_views][None],
                                                    focal, z_near, z_far)[0]  # (len(target_views), 16, H, W)
                # 2. use feature images and denoiser to generate novel views
                steps = int(kwargs["denoising_steps"])
                sampler_fn = edm_sampler if steps > 1 else one_step_edm_sampler
                novel_images = sampler_fn(net.denoiser, latents.expand(len(target_views), -1, -1, -1), feature_images, None, num_steps=steps)  # (len(target_views), 3, H, W)
                novel_images = torch.clamp(novel_images, min=-1, max=1).cpu()  # (len(target_views), 3, H, W)
                # 3. evaluate and save images
                gt_images = images[target_views]
                for j in range(novel_images.shape[0]):
                    pred = novel_images[j]  # (3, H, W)
                    gt = gt_images[j]  # (3, H, W)
                    pred = img_tensor_to_npy(pred)  # (H, W, 3)
                    gt = img_tensor_to_npy(gt)  # (H, W, 3)
                    target_view_id = target_views[j].item()
                    # get ssim and psnr
                    cur_ssim = skimage.metrics.structural_similarity(
                        pred,
                        gt,
                        multichannel=True,
                        data_range=255,
                    )
                    cur_psnr = skimage.metrics.peak_signal_noise_ratio(
                        pred,
                        gt,
                        data_range=255
                    )
                    ssims.append(cur_ssim)
                    psnrs.append(cur_psnr)
                    # save images
                    imageio.imwrite(os.path.join(kwargs["out_dir"], obj_name, f"{target_view_id:0>6}.png"),
                                    pred)

            # record and print metrics
            ssim = sum(ssims) / len(ssims)
            psnr = sum(psnrs) / len(psnrs)
            total_psnr += psnr
            total_ssim += ssim
            cnt += 1
            print(
                "curr psnr",
                psnr,
                "ssim",
                ssim,
                "running psnr",
                total_psnr / cnt,
                "running ssim",
                total_ssim / cnt,
            )
            finish_file.write(
                "{} {} {}\n".format(obj_name, psnr, ssim)
            )
    
    # final record and print
    print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
    finish_file.write(f"final_psnr {total_psnr / cnt} final_ssim {total_ssim / cnt}\n")
    finish_file.close()

if __name__ == "__main__":
    main()
