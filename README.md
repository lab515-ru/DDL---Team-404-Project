# Multi-view people localization on Perlmutter (Team 404)

## Setup
1. Ask Leana for read access to the project folder.
2. `source env.sh` (activates the environment, sets $P and $OUT).
3. Your outputs go under $OUT. Never write into $P.

## Data (read-only, under $P/data)
- wildtrack: 7 cams, 400 frames, cache wt_270x480.npy, target grid 360x120 (rows=Y, cols=X), calibration in cm
- multiviewx: 6 cams, 400 frames, cache mx_270x480.npy, target grid 160x250, calibration in meters
- Do not mix the two in one batch yet. Do not commit data, weights or checkpoints.
- EPFL-RLC and MMPTRACK are not shared (terms not yet checked).

## Run (4 GPUs, one node)
    salloc -A m4431_g -C gpu -q debug -N 1 -t 00:10:00 --gpus-per-node=4
    torchrun --standalone --nproc-per-node=4 train_fsdp.py --dataset real \
      --data_root $P/data/wildtrack/Wildtrack_dataset --cache $P/data/wildtrack/wt_270x480.npy \
      --views 7 --img_h 270 --img_w 480 --grid_h 360 --grid_w 120 --batch_per_gpu 8 --steps 100 \
      --pretrained_path $P/weights/vit_base_patch16_224.pth --run_name NAME_test \
      --log_dir $OUT/logs --ckpt_dir $OUT/ckpt/NAME_test --node_hours_csv $OUT/node_hours.csv
    exit

Resume = rerun the same command (--steps is the total target). Results: logs/<run>/summary.json.

## Rules
- Always set -t on salloc, and `exit` when done (hostname nid... = clock running).
- Use unique --run_name. Delete old checkpoints.
- Loss/MFU are not results yet (fusion is a placeholder).
- Real node-hours: iris.nersc.gov, not node_hours.csv.
