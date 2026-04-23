# Neuromorphic-projects
Improving Latency in Bird Detection on Edge  Devices
# ReviewKD for YOLO26x -> YOLO26n (Ultralytics-style)

Files:
- 'Precompute_teacher_features.py': Load YOLO26-X teacher, freeze it (it won't learn anymore).
- `reviewkd_modules.py`: ABF, HCL, feature hooks, channel inference.
- `Offline_reviewkd_trainer.py`: custom DetectionModel + DetectionTrainer for ReviewKD.
- `train_offline.py`: example training entrypoint.

## Where to place these files

Copy the three `.py` files into the root of your YOLO26 repo (same place you run training from), then edit the paths in `train_offline.py`.

## Basic workflow
--- Run Precompute_teacher_features, it will run for every training image, the big teacher once, and save the teacher's intermediate "thoughts" to disk.
1. Teacher: Use your trained YOLO26x bird checkpoint.
2. Student init: use your trained YOLO26n bird checkpoint.
3. Run `python train_offline.py` on the HPC.
4. Validate the new checkpoint.
5. Compare to your plain YOLO26n baseline.

## Notes

- This code distills the multi-scale neck features that feed the Detect head.
- If your YOLO26 build differs from the standard Ultralytics graph, you may need to set `kd_hook_indices=[...]` manually.
- The goal is to narrow the gap to the teacher while keeping YOLO26n latency unchanged. Exact teacher-level accuracy is not guaranteed.
