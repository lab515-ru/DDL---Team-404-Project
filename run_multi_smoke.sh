D=$P/data
torchrun --standalone --nproc-per-node=4 train_fsdp.py --dataset real --multi --domains all \
  --mmp_cache $D/mmptrack/cache --wt_root $D/wildtrack/Wildtrack_dataset --wt_cache $D/wildtrack/wt_270x480.npy \
  --mx_root $D/multiviewx/MultiviewX --mx_cache $D/multiviewx/mx_270x480.npy \
  --fusion calibrated --pos_weight 300 --img_h 270 --img_w 480 --batch_per_gpu 8 --workers 8 \
  --steps 400 --ckpt_every 100000 --eval_thr 0.3 0.5 0.7 --val_stride 4 \
  --pretrained_path $P/weights/vit_base_patch16_224.pth \
  --run_name multi_smoke --log_dir $P/logs --ckpt_dir $P/ckpt/ms --node_hours_csv $P/node_hours.csv
