# written by JhihYang Wu <jhihyangwu@arizona.edu>
# prep a eval out folder for getting FID score
# usage: python prep_eval_fid.py path_to_folder_with_finish.txt
# does nothing if already ran before

import sys
import os
from tqdm import tqdm

GPU_ID = 0
GT_DATA_PATH = "/workspace/data/srnchairs/test"

def main():
    # get imgs to test on
    path = get_eval_out_path()
    scene_names = get_scene_names(path)
    # check if we even need to copy images
    path1 = path + "_tmp_fid"
    path2 = os.path.join(path1, "pred")
    path3 = os.path.join(path1, "ground")
    final_cmd = f"python -m pytorch_fid --device cuda:{GPU_ID} {path2} {path3}"
    if os.path.exists(path1):
        print("Please run")
        print(final_cmd)
        sys.exit(0)
    # make folders
    os.makedirs(path1, exist_ok=False)
    os.makedirs(path2, exist_ok=False)
    os.makedirs(path3, exist_ok=False)
    # put all pred and ground in different folders
    count = 0
    for scene_name in tqdm(scene_names):
        img_filenames = os.listdir(os.path.join(path, scene_name))
        for filename in img_filenames:
            assert filename.endswith(".png")
            os.system(f"cp {os.path.join(path, scene_name, filename)} {os.path.join(path2, f'{count}.png')}")
            os.system(f"cp {os.path.join(GT_DATA_PATH, scene_name, filename)} {os.path.join(path3, f'{count}.png')}")
            count += 1
    # after images are copied
    print("Please run")
    print(final_cmd)

def get_eval_out_path():
    if len(sys.argv) != 2:
        print("Usage: python prep_eval_fid.py path_to_folder_with_finish.txt")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(os.path.join(path, "finish.txt")):
        print(f"Bad path {path}")
        sys.exit(1)
    return path if path[-1] != "/" else path[:-1]

def get_scene_names(path):
    scene_names = []
    with open(os.path.join(path, "finish.txt")) as file:
        for line in file:
            line = line.split(" ")
            if len(line) == 3:
                scene_names.append(line[0])
            else:
                assert line[0] == "final_psnr"
                return scene_names
    print("Bad finish.txt")
    sys.exit(1)

if __name__ == "__main__":
    main()
