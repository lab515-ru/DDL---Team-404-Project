source args.sh
C=$P/data/mmptrack/cache
TR=$(ls $C/train__*office_*.json | xargs -n1 basename | sed 's/.json//' | paste -sd,)
VA=$(ls $C/validation__*office_*.json | xargs -n1 basename | sed 's/.json//' | paste -sd,)
echo $TR; echo $VA
for W in 30 100 300; do
  torchrun --standalone --nproc-per-node=4 train_fsdp.py $ARGS --real_set mmptrack --data_root $C \
    --mmp_scene office --mmp_train $TR --mmp_val $VA --views 5 --grid_h 60 --grid_w 67 \
    --steps 600 --pos_weight $W --eval_thr 0.05 0.3 0.5 0.7 0.9 0.99 \
    --run_name off_w${W} --ckpt_dir $P/ckpt/off > /dev/null 2>&1
  rm -rf $P/ckpt/off
  echo done $W
done
