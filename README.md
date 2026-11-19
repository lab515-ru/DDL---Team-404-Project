# Multi-view person localization on Perlmutter (Team 404)

Distributed-training study with a measured accuracy result. Several fixed, overlapping cameras ->
shared ImageNet-pretrained ViT encoder -> calibrated projection of image features onto a top-down
ground grid -> person-location map. Final deliverable: one model trained on all datasets
(MMPTRACK 5 scenes, WILDTRACK, MultiviewX) and tested on EPFL-RLC, with a scaling, memory and
profiling study on Perlmutter (FSDP, 1/4/8+ GPUs).

Team: [names / roles]     Repo: github.com/lab515-ru/DDL---Team-404-Project

## 1. Where everything is
Project root `$P` = /pscratch/sd/l/leana_ab/perlmutter-class/project (scratch: not backed up).

| Path | Contents |
|---|---|
| code/ | all scripts (this repo) |
| data/wildtrack, data/multiviewx | extracted datasets + packed caches `wt_270x480.npy`, `mx_270x480.npy` |
| data/mmptrack | MMPTRACK zips, `cache/` (20 packed recordings, 18 GB), `calib_test/` (test frames) |
| weights/ | ImageNet ViT-B weights (`vit_base_patch16_224.pth`) |
| logs/<run>/ | `summary.json` (timing, throughput, memory, NCCL), `eval.json` (accuracy) |
| Environment | /pscratch/sd/l/leana_ab/perlmutter-class/dist_dl (torch 2.13+cu132, timm 1.0.30, scipy) |

Slurm account `m4431_g`, billed in node-hours (1 node = 4 A100). Compute nodes have no internet.
Datasets are NOT in git (see section 2 for terms). Checkpoints are not kept.

### Code map
| File | Role |
|---|---|
| train_fsdp.py | FSDP bf16 training, sharded checkpoints/resume, profiler, held-out evaluator |
| geometry.py | calibrated ground-to-feature sampling grids for WILDTRACK / MultiviewX |
| mmptrack.py, cache_mmp.py | MMPTRACK per-scene geometry, dataset, cache builder |
| wildtrack.py, multiviewx.py | loaders and calibration readers |
| cache_wildtrack.py, cache_mx.py | pack resized frames into one array (removes data-loading bottleneck) |
| baseline_meanmap.py, baseline_mmp.py | "average training map" baselines |
| test_*.py | geometry checks (project labeled people into cameras, check against boxes) |
| run_th.sh, sum_th.py | 5-scene x 3-repeat threshold sweep and summary |
| run_pw.sh, sum_pw.py, run_office.sh, run_scenes.sh | pos_weight and per-scene runs |
| epfl_rlc.py | EPFL-RLC grid/geometry from its positions file, GIF demo, per-cell scorer (not run) |
| args.sh, env.sh | shared run arguments / environment |
| domains.py | domain registry (camera count, grid, geometry, splits per dataset/scene) and domain-grouped batch sampler |
| check_vivit.py | downloads/checks ViViT weights for the temporal experiment |

## 2. Datasets, calibration, conventions
Every dataset's calibration was verified the same way: project each labeled person's ground
position into each camera and check it lands at the labeled box.

| Dataset | Cameras | Labeled data used | Calibration conventions | Check |
|---|---|---|---|---|
| WILDTRACK | 7, 1920x1080 | 400 frames | OpenCV rvec/tvec + `intrinsic_zero` (undistorted). World in **cm**. Camera order CVLab1-4, IDIAP1-3 = C1-C7. Grid 480x1440, 2.5 cm, origin (-300,-900) cm | 149/149 boxes; fusion geometry 138/138 |
| MultiviewX | 6, 1920x1080 | 400 frames | World in **meters**; x=(ID%1000)*0.025, y=(ID//1000)*0.025. **Camera 4 has lens distortion** (k1,k2,p1,p2,k3 applied in geometry.py). **Camera depth is negative** (use -z for visibility) | 947/947 with distortion (919/947 without); fusion geometry 926/926 |
| MMPTRACK | 4 (cafe, industry, lobby), 5 (office), 6 (retail); 640x360 | official train and validation (different people, same scenes) | JSON per scene, world in **mm**. Camera point = R^T(X - t). CSV label (x, y): **U = MinU + 20*y, V = MinV + 20*x**, foot at W=0. Train and validation calibrations are identical | 5 scenes, about 92-97% on 3 frames each (earlier test); loader test: 84/84, 81/84, 78/82, 103/104, 113/114 |
| EPFL-RLC | 3, 480x270 | 4,088 occupied/free cell labels | Tsai model not converted; instead uses its `positions_55x45.json` (pixel box per cell per view). Grid inferred as 45 rows x 55 cols | `check` and demo GIF written; **not scored** |

Model-side shapes: images resized to 270x480, ViT feature grid 16x30 (bottom 14 px cropped).
Target grids use 10 cm cells so the evaluator's match radius is always 0.5 m (5 cells):
WILDTRACK 360x120, MultiviewX 160x250, MMPTRACK per scene (rows x cols):
cafe_shop 74x59, industry_safety 62x71, lobby 76x73, office 60x67, retail 66x85.
Cameras that cannot see a ground cell are masked, so any camera count works, but `--views`
must equal the scene's camera count and one run uses one dataset/scene (grids and camera counts
differ).

Licensing/terms: MultiviewX CC BY-SA 4.0 (Hou et al., ECCV 2020). EPFL-RLC research/evaluation
only. WILDTRACK: cite Chavdarova et al. MMPTRACK: research use; data is not redistributed here.
No dataset files, weights or checkpoints are in git.

## 3. How to run
    source env.sh
    salloc -A m4431_g -C gpu -q debug -N 1 -t 00:10:00 --gpus-per-node=4   # hostname must start with nid
    source args.sh
    torchrun --standalone --nproc-per-node=4 train_fsdp.py $ARGS --run_name NAME --ckpt_dir DIR
    exit                                                                  # stops the billing clock
Two nodes: `srun --ntasks-per-node=1 torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d
--rdzv-endpoint=$(hostname):29500 train_fsdp.py $ARGS ...`

MMPTRACK scene (example, cafe_shop): build the cache with
`python cache_mmp.py --zip training.zip --split train --scene cafe_shop --n 2 --every 10 --out cache`
(and `--split validation`), then run `bash run_th.sh cafe_shop` (3 repeats, thresholds 0.05-0.9),
then `python sum_th.py`.

Checkpoints are sharded (~1 GB each) and valid only for the code that wrote them. Resume = rerun
the same command (`--steps` is the total target). Resume was tested at the same GPU count only.

## 4. Results

### 4a. Scaling (WILDTRACK, 7 views, ViT-B, 8 samples/GPU, cached data, calibrated fusion, 200 steps)
Weak scaling (global batch grows with GPU count). **Single run per point.**

| | 1 GPU | 4 GPUs (1 node) | 8 GPUs (2 nodes) |
|---|---|---|---|
| Throughput (samples/s) | 57.6 | 222.6 | 380.1 |
| Efficiency vs 1 GPU | 100% | 97% | 82% |
| Data-wait fraction | 4% | 19% | 23% |
| Profiled NCCL time/step | 0 | 0.03 s | 0.31 s |

- 4-GPU throughput across three runs: 216-233 samples/s (about +-3%), so treat the efficiency
  figures as +-a few points. [Replace with repeat means if repeated.]
- Microbenchmark all-reduce bus bandwidth: about 170-190 GB/s inside a node, about 4.6 GB/s between nodes.
- Packing resized frames into one array cut data wait from 35% to 4.5% (measured with the earlier
  placeholder fusion, same node, back to back).
- Replacing the placeholder camera-average with calibrated fusion made steps about 4x faster, so
  scaling numbers from the placeholder are retired.
- FSDP saves little memory at ViT-B size (0.34 vs 1.38 GB of state per GPU at 4 GPUs).
- MMPTRACK runs (4 cameras, 4 GPUs): 377-389 samples/s, 5.2 GB peak per GPU.

### 4b. Accuracy
Metric: precision/recall/F1 from our own matching (predicted peaks within 0.5 m of true positions,
one-to-one). This is NOT MODA and is not comparable to published MODA values.
Baseline = average training map, told the true person count per frame.

**MMPTRACK, one model per scene**, `--pos_weight 300`, 600 steps, 4 GPUs, trained on 2 recordings,
evaluated on 2 recordings of different people (every 10th frame), threshold 0.5, 3 repeats:

| Scene | Cameras | Baseline F1 | Model F1 (sd) | Precision | Recall |
|---|---|---|---|---|---|
| cafe_shop | 4 | 0.320 | 0.812 (0.013) | 0.885 | 0.751 |
| industry_safety | 4 | 0.323 | 0.841 (0.008) | 0.852 | 0.832 |
| lobby | 4 | 0.371 | 0.845 (0.011) | 0.839 | 0.856 |
| office | 5 | 0.470 | 0.915 (0.015) | 0.931 | 0.901 |
| retail | 6 | 0.232 | 0.848 (0.005) | 0.944 | 0.770 |
| **Mean of five** | | 0.343 | **0.852** | | |

**WILDTRACK**, last 40 frames held out (5-frame gap), `--pos_weight 100`, 600 steps, **single run**:
F1 0.746 at threshold 0.5 (precision 0.766, recall 0.727); baseline 0.249.

### 4c. Findings worth knowing
- **Threshold:** the weighted loss inflates probabilities. In office, thresholds 0.05-0.4 give
  about 250k false positives (F1 about 0.05) and 0.5 and above give F1 0.915. The other scenes
  are insensitive (within about 0.03). One threshold of 0.5 works for all five scenes. It was
  chosen on the same validation recordings that are scored, so reported F1 is slightly optimistic.
- **pos_weight** (cafe_shop, threshold 0.05, 3 repeats): 30 -> 0.724, 100 -> 0.812, 300 -> 0.833.
  1000 was not tried.
- **Run-to-run noise:** seeding (`torch.manual_seed(0)`) does not make runs identical; two
  identical seeded runs differed by 0.03 F1 at the lower weighting. Always compare means of repeats.

## 5. Final-phase plan (status: planned unless marked done)

Splits (all data used, no change between baseline and comparisons):
| Dataset | Train | Validation | Test |
|---|---|---|---|
| MMPTRACK, 5 scenes | 44 official train recordings, every 5th frame | 12 of the 24 official validation recordings (split by recording) | the other 12 |
| WILDTRACK | all but last 40 frames (5-frame gap) | last 40 frames | none |
| MultiviewX | all but last 40 frames (5-frame gap) | last 40 frames | none |
| EPFL-RLC | none | none | all 4,088 labeled cells (per-cell AUC/accuracy; grid spacing unknown) |
Each step draws one dataset/scene (same on all GPUs) so camera counts never mix in a batch;
sampling weights are sqrt(training size). One decision threshold for all domains, chosen on
validation, fixed before the test is touched.

Experiments (2 seeds each; ViT-L full fine-tuning under FSDP on 4 GPUs is the baseline):
1. Baseline ViT-L on all data, with a bird's-eye view demo (cameras, fields of view, person circles
   with confidence, colour by number of cameras seeing the cell)
2. Scaling sweep 1/4/8 GPUs (and more nodes), weak and strong, with NCCL and Nsight profiling
3. ViT-B vs ViT-L
4. DDP vs FSDP memory test (OOM only if it occurs; LoRA is out of scope)
5. Activation-checkpointing memory/speed ablation
6. Evaluation on varying camera counts (cameras masked at test time)
7. ViViT as a separate temporal experiment (MMPTRACK only, clips of earlier frames vs a single-frame
   control and an averaged-frames control)
8. Edge-vs-central bandwidth/latency benchmark (`--bench_only`, untested)
Possible extras: longer training, data-volume curve, camera-dropout training, resolution study.
Compute budget: about 100 node-hours team total; planned work about 14-25 node-hours before extras
(ViT-L throughput is an estimate from ViT-B until benchmarked).

## 6. Status and limitations
Done and measured: calibration for all datasets; ViT-B scaling (single run per point, weak scaling);
per-scene MMPTRACK models (F1 0.81-0.92 vs baselines 0.23-0.47); WILDTRACK F1 0.75 (one run).
Not done (planned in section 5): ViT-L, multi-dataset training, seeds/repeats for scaling,
strong scaling, cross-site test, EPFL-RLC scoring, ViViT, DDP/checkpointing ablations,
bird's-eye view, resume at a different GPU count.
Known limitations: our own P/R/F1 (not MODA); threshold chosen on reported validation data;
validation shares scene and cameras with training (different people only) until the test split
is used; EPFL-RLC labels are per-cell only; boxes assume full standing height; ~7% of WILDTRACK
person-views fall outside the sampled feature map; MultiviewX has never been trained.