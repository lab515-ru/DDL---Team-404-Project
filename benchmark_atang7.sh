#!/bin/bash
#SBATCH --job-name=atang7-fsdp
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=/pscratch/sd/a/atang7/ddl_project/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/a/atang7/ddl_project/logs/%x-%j.err

set -e
source /pscratch/sd/l/leana_ab/perlmutter-class/dist_dl/bin/activate
cd /pscratch/sd/l/leana_ab/perlmutter-class/project/code

srun torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  train_fsdp.py \
  --model vit_base_patch16_224 \
  --dataset synthetic \
  --steps 50 \
  --run_name atang7_fsdp_benchmark \
  --log_dir /pscratch/sd/a/atang7/ddl_project/logs \
  --ckpt_dir /pscratch/sd/a/atang7/ddl_project/checkpoints \
  --node_hours_csv /pscratch/sd/a/atang7/ddl_project/logs/node_hours.csv
