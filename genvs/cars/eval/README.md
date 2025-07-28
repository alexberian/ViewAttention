### Files and Purpose
```
eval.py -- evaluates trained GeNVS
eval.sh -- shell script to run eval.py easier
eval_all_cam_weighting.py -- tests out all the camera weighting algorithms by repeatedly running eval.py
eval_all_cam_weighting_other_metrics.py -- after running eval_all_cam_weighting.py, run this to get fid, lpips, and dists scores
eval_all_cam_weighting_csv.py -- generates a csv file with all the camera weighting results

prep_eval_fid.py -- prep a folder for getting FID score, you shouldn't need to manually run this
eval_fid_lpips_dists.py -- given the path to a folder with finish.txt, it will generate finish_2.txt and finish_3.txt with FID, LPIPS, and DISTS scores
```
