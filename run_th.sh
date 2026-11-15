source args.sh
C=$P/data/mmptrack/cache
run() {
  S=$1; V=$2; GH=$3; GW=$4
  TR=$(ls $C/train__*${S}_*.json | xargs -n1 basename | sed 's/.json//' | paste -sd,)
  VA=$(ls $C/validation__*${S}_*.json | xargs -n1 basename | sed 's/.json//' | paste -sd,)
  echo $S $TR $VA
  for i in 1 2 3; do
    torchrun --standalone --nproc-per-node=4 train_fsdp.py $ARGS --real_set mmptrack --data_root $C \
      --mmp_scene $S --mmp_train $TR --mmp_val $VA --views $V --grid_h $GH --grid_w $GW \
      --steps 600 --pos_weight 300 --eval_thr 0.05 0.2 0.3 0.4 0.5 0.6 0.7 0.9 \
      --run_name th_${S}_r$i --ckpt_dir $P/ckpt/th > /dev/null 2>&1
    rm -rf $P/ckpt/th
    echo done $S r$i
  done
}
for X in "$@"; do
  case $X in
    cafe_shop) run cafe_shop 4 74 59;;
    industry_safety) run industry_safety 4 62 71;;
    lobby) run lobby 4 76 73;;
    office) run office 5 60 67;;
    retail) run retail 6 66 85;;
  esac
done
