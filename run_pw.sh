source args.sh
C=$P/data/mmptrack/cache
TR=$(ls $C/train__*cafe_shop*.json | xargs -n1 basename | sed 's/.json//' | paste -sd,)
VA=$(ls $C/validation__*cafe_shop*.json | xargs -n1 basename | sed 's/.json//' | paste -sd,)
echo $TR; echo $VA
for W in 30 100 300; do
  for i in 1 2 3; do
    torchrun --standalone --nproc-per-node=4 train_fsdp.py $ARGS --real_set mmptrack --data_root $C \
      --mmp_scene cafe_shop --mmp_train $TR --mmp_val $VA --views 4 --grid_h 74 --grid_w 59 \
      --steps 600 --pos_weight $W --eval_thr 0.05 0.1 0.2 \
      --run_name pw${W}_r$i --ckpt_dir $P/ckpt/pw > /dev/null 2>&1
    rm -rf $P/ckpt/pw
    echo done pw$W r$i
  done
done
