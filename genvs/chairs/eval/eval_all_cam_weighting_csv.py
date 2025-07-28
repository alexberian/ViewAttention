# written by JhihYang Wu <jhihyangwu@arizona.edu>
# generates a csv file with all the camera weighting results
# to run simply execute: python eval_all_cam_weighting_csv.py

from eval_all_cam_weighting import OUT_DIR
import os
import csv

def main():
    with open("cam_weighting.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["directory", "fid", "lpips", "dists", "psnr", "ssim"])
        for path in sorted(os.listdir(OUT_DIR)):
            if path.endswith("tmp_fid"):
                continue  # not a result directory so skip
            path = os.path.join(OUT_DIR, path)  # get full path
            path1 = os.path.join(path, "finish_2.txt")
            path2 = os.path.join(path, "finish_3.txt")
            if os.path.exists(path1) and os.path.exists(path2):
                with open(path1, "r") as file:
                    final_line = file.readlines()[-1]
                    assert final_line.startswith("final psnr: ")
                    _, _, psnr, _, _, ssim, _, _, lpips, _, _, dists = final_line.split()
                with open(path2, "r") as file:
                    final_line = file.readlines()[-1]
                    assert final_line.startswith("FID: ")
                    fid = final_line.split()[-1]
            csv_writer.writerow([path, fid, lpips, dists, psnr, ssim])

if __name__ == "__main__":
    main()
