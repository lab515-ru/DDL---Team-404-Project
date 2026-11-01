ARGS="--dataset real --real_set wildtrack --data_root $P/data/wildtrack/Wildtrack_dataset
 --cache $P/data/wildtrack/wt_270x480.npy --fusion calibrated --pos_weight 100
 --views 7 --img_h 270 --img_w 480 --grid_h 360 --grid_w 120
 --batch_per_gpu 8 --workers 8 --steps 200 --ckpt_every 1000
 --prof_start 40 --prof_steps 3 --stats_skip 10 --log_every 10
 --pretrained_path $P/weights/vit_base_patch16_224.pth
 --log_dir $P/logs --node_hours_csv $P/node_hours.csv"
