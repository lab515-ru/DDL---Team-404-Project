#!/usr/bin/env python3
"""
Multi-view ViT people-localization training with FSDP, plus performance tracking
(latency, bandwidth, memory/compute-bound indicators, node-hours) for later comparison.

Launch (per node, inside an sbatch job):
    srun --ntasks-per-node=1 torchrun --nnodes=$SLURM_NNODES --nproc-per-node=4 \
         --rdzv-backend=c10d --rdzv-endpoint=$(hostname):29500 \
         train_fsdp.py --model vit_base_patch16_224 --steps 500 --run_name vitb_fsdp_1n

    sbatch hints:  #SBATCH --requeue   #SBATCH --signal=B:USR1@120
    (the script saves a checkpoint on SIGUSR1 and exits; rerun the same command to resume)

Edge-vs-central latency / bandwidth benchmark (single GPU, no training):
    python train_fsdp.py --bench_only --model vit_base_patch16_224 --views 4 --fps 10

What it records (rank 0, in <log_dir>/<run_name>/):
    metrics.jsonl   per-step: data / forward / backward / optimizer / total time, loss,
                    samples/s, peak memory, MFU
    summary.json    averages, p50/p95 step latency, NCCL kernel time per step (profiled window),
                    estimated communication volume and effective bandwidth, NCCL microbenchmark
                    (all-reduce / all-gather / reduce-scatter bus bandwidth), GPU and memory-
                    controller utilization (pynvml), estimated DDP-equivalent comm and memory,
                    node-hours for this segment
    <node_hours_csv> one appended row per job segment (shared tracker)

Caveats: see the notes next to NotImplementedError (real data) and the fuse() stub (calibrated
ground-plane projection). Requires torch >= 2.2 and timm. Perlmutter compute nodes usually have no
internet, so download pretrained weights on a login node and pass --pretrained_path.
"""
import argparse
import csv
import getpass
import json
import os
import signal
import threading
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import timm
from timm.models.vision_transformer import Block

STOP_FLAG = {"stop": False}


# --------------------------------------------------------------------------- args
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--fusion", default="mean", choices=["mean", "calibrated"])
    p.add_argument("--real_set", default="wildtrack", choices=["wildtrack", "multiviewx"])
    p.add_argument("--cache", default=None, help="packed .npy from cache_wildtrack.py")
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--pretrained_path", default=None, help="local timm weights file")
    p.add_argument("--img_h", type=int, default=224)
    p.add_argument("--img_w", type=int, default=384)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--grid_h", type=int, default=80, help="ground-plane map height")
    p.add_argument("--grid_w", type=int, default=120, help="ground-plane map width")
    p.add_argument("--proj_ch", type=int, default=64)
    p.add_argument("--batch_per_gpu", type=int, default=2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--activation_ckpt", action="store_true")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "real"])
    p.add_argument("--data_root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--run_name", default="run")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--ckpt_dir", default="ckpt")
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--stats_skip", type=int, default=10, help="warmup steps ignored in stats")
    p.add_argument("--prof_start", type=int, default=20)
    p.add_argument("--prof_steps", type=int, default=5)
    p.add_argument("--peak_tflops", type=float, default=312.0, help="A100 bf16 dense")
    p.add_argument("--node_hours_csv", default="node_hours.csv")
    p.add_argument("--bench_only", action="store_true")
    p.add_argument("--fps", type=float, default=10.0, help="camera frame rate for bandwidth calc")
    return p.parse_args()


# --------------------------------------------------------------------------- model
class MultiViewViT(nn.Module):
    """Shared ViT encoder per camera -> 1x1 projection -> fuse views -> ground-plane head."""

    def __init__(self, a):
        super().__init__()
        kw = dict(pretrained=False, num_classes=0, img_size=(a.img_h, a.img_w))
        if a.pretrained_path:
            kw.update(pretrained=True, pretrained_cfg_overlay=dict(file=a.pretrained_path))
        self.encoder = timm.create_model(a.model, **kw)
        self.gh, self.gw = self.encoder.patch_embed.grid_size
        d = self.encoder.embed_dim
        self.proj = nn.Conv2d(d, a.proj_ch, 1)
        self.head = nn.Sequential(
            nn.Conv2d(a.proj_ch, a.proj_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(a.proj_ch, 1, 1))
        self.grid = (a.grid_h, a.grid_w)
        self.geo = None
        if getattr(a, "fusion", "mean") == "calibrated":
            from geometry import make_geometry
            self.geo = make_geometry(a.real_set, a.data_root, (a.img_h, a.img_w),
                                     (a.grid_h, a.grid_w), (self.gh, self.gw))

    def encode(self, x):  # x: (N,3,H,W) -> (N,C,gh,gw)
        tok = self.encoder.forward_features(x)
        tok = tok[:, self.encoder.num_prefix_tokens:]
        n, _, d = tok.shape
        return self.proj(tok.transpose(1, 2).reshape(n, d, self.gh, self.gw))

    def fuse(self, f, b, v):
        if self.geo is None:   # old placeholder
            f = f.view(b, v, *f.shape[1:]).mean(1)
            f = F.interpolate(f, size=self.grid, mode="bilinear", align_corners=False)
            return self.head(f)
        grid, mask = self.geo
        dt = f.dtype
        grid, mask = grid.to(f.device), mask.to(f.device)
        s = F.grid_sample(f.float(), grid.repeat(b, 1, 1, 1), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
        m = mask.repeat(b, 1, 1).unsqueeze(1)
        s = (s * m).view(b, v, *s.shape[1:])
        m = m.view(b, v, *m.shape[1:])
        out = s.sum(1) / m.sum(1).clamp(min=1)
        return self.head(out.to(dt))

    def forward(self, views):  # (B,V,3,H,W)
        b, v = views.shape[:2]
        return self.fuse(self.encode(views.flatten(0, 1)), b, v)


# --------------------------------------------------------------------------- data
class SyntheticMultiView(Dataset):
    def __init__(self, a, n=4096):
        self.a, self.n = a, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        a = self.a
        views = torch.randn(a.views, 3, a.img_h, a.img_w, generator=g)
        target = (torch.rand(1, a.grid_h, a.grid_w, generator=g) > 0.98).float()
        return views, target


class RealMultiView(Dataset):
    def __init__(self, a):
        if a.real_set == "multiviewx":
            from multiviewx import MultiviewXDataset
            self.ds = MultiviewXDataset(a.data_root, img_hw=(a.img_h, a.img_w),
                                        grid_hw=(a.grid_h, a.grid_w), cache=a.cache)
        else:
            from wildtrack import WildtrackDataset
            self.ds = WildtrackDataset(a.data_root, img_hw=(a.img_h, a.img_w),
                                       grid_hw=(a.grid_h, a.grid_w), cache=a.cache)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


# --------------------------------------------------------------------------- utils
def setup_dist():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return rank, world, local


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


class GpuSampler(threading.Thread):
    """Samples GPU util % and memory-controller util % (rough memory-bound indicator)."""

    def __init__(self, idx, period=0.5):
        super().__init__(daemon=True)
        self.ok, self.u, self.m, self.period = False, [], [], period
        self.evt = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(idx)
            self.ok = True
        except Exception:
            pass

    def run(self):
        while self.ok and not self.evt.is_set():
            r = self.nv.nvmlDeviceGetUtilizationRates(self.h)
            self.u.append(r.gpu)
            self.m.append(r.memory)
            time.sleep(self.period)


def nccl_time_s(prof):
    """Sum of NCCL kernel device time in the profiled window (includes waiting on peers)."""
    tot = 0.0
    for e in prof.key_averages():
        if "nccl" in e.key.lower():
            t = getattr(e, "self_device_time_total", None)
            if t is None:
                t = getattr(e, "self_cuda_time_total", 0)
            tot += t
    return tot / 1e6


def nccl_microbench(world, size_mb=256, iters=20):
    if world == 1:
        return {}
    n = size_mb * 1024 * 1024 // 2
    size_b = n * 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    shard = torch.ones(n // world, dtype=torch.bfloat16, device="cuda")
    full = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    out_shard = torch.empty(n // world, dtype=torch.bfloat16, device="cuda")

    def run(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters

    ar = run(lambda: dist.all_reduce(x))
    ag = run(lambda: dist.all_gather_into_tensor(full, shard))
    rs = run(lambda: dist.reduce_scatter_tensor(out_shard, full))
    f = (world - 1) / world
    return {
        "msg_MB": size_mb,
        "allreduce_ms": ar * 1e3, "allreduce_busbw_GBps": 2 * f * size_b / ar / 1e9,
        "allgather_ms": ag * 1e3, "allgather_busbw_GBps": f * size_b / ag / 1e9,
        "reducescatter_ms": rs * 1e3, "reducescatter_busbw_GBps": f * size_b / rs / 1e9,
    }


def init_optim_state(opt):
    """Adam state is lazy; create it so a checkpoint can be loaded into it."""
    lrs = [g["lr"] for g in opt.param_groups]
    for g in opt.param_groups:
        g["lr"] = 0.0
        for p in g["params"]:
            p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)
    for g, lr in zip(opt.param_groups, lrs):
        g["lr"] = lr


def latest_ckpt(ckpt_dir):
    f = Path(ckpt_dir) / "latest"
    if not f.exists():
        return None, 0
    name = f.read_text().strip()
    return Path(ckpt_dir) / name, int(name.split("_")[1])


def save_ckpt(model, opt, step, ckpt_dir, rank):
    path = Path(ckpt_dir) / f"step_{step:07d}"
    msd, osd = get_state_dict(model, opt)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(path))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        (Path(ckpt_dir) / "latest").write_text(path.name)


def load_ckpt(model, opt, path):
    msd, osd = get_state_dict(model, opt)
    pg = osd.pop("param_groups", None)
    sd = {"model": msd, "optim": osd}
    dcp.load(sd, checkpoint_id=str(path))
    if pg is not None:
        sd["optim"]["param_groups"] = pg
    set_state_dict(model, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])


# --------------------------------------------------------------------------- edge vs central
@torch.no_grad()
def edge_central_bench(a):
    """Latency and bytes for: camera-side encoder -> network -> central fusion.
    Latency does not depend on weights, so an untrained model is used."""
    dev, bf = "cuda", torch.bfloat16
    model = MultiViewViT(a).to(dev, bf).eval()
    H, W, V = a.img_h, a.img_w, a.views

    def timeit(fn, n=50, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1e3

    one = torch.randn(1, 3, H, W, device=dev, dtype=bf)
    enc_ms = timeit(lambda: model.encode(one))
    feats = model.encode(torch.randn(V, 3, H, W, device=dev, dtype=bf))
    fuse_ms = timeit(lambda: model.fuse(feats, 1, V))
    n_tok = model.gh * model.gw
    d = model.encoder.embed_dim
    sizes = {  # bytes per camera frame, by what you send
        "raw_frame_uint8": 3 * H * W,
        "encoder_tokens_bf16": (n_tok + 1) * d * 2,
        "projected_features_bf16": a.proj_ch * n_tok * 2,
        "detections_only_approx": 64,  # a few coordinates per person; edit as needed
    }
    report = {"encoder_ms_per_camera": enc_ms, "fusion_ms_for_V_views": fuse_ms,
              "views": V, "fps": a.fps, "bytes_per_frame": sizes, "needed_Mbps": {},
              "transfer_ms_per_frame_set": {}}
    for k, b in sizes.items():
        report["needed_Mbps"][k] = b * V * a.fps * 8 / 1e6
        report["transfer_ms_per_frame_set"][k] = {
            f"{g}Gbps": b * V * 8 / (g * 1e9) * 1e3 for g in (0.1, 1, 10)}
    out = Path(a.log_dir) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "edge_central.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------- main
def main():
    a = parse()
    if a.bench_only:
        edge_central_bench(a)
        return

    rank, world, local = setup_dist()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    nnodes = max(1, world // local_world)
    out = Path(a.log_dir) / a.run_name
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        Path(a.ckpt_dir).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGUSR1, lambda *_: STOP_FLAG.update(stop=True))

    # model
    model = MultiViewViT(a)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    model = model.cuda()
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)
    model = FSDP(
        model, sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=mp,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
        device_id=local, use_orig_params=True)
    if a.activation_ckpt:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing, checkpoint_wrapper)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper,
                                       check_fn=lambda m: isinstance(m, Block))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup))

    # resume
    start = 0
    ck, start = latest_ckpt(a.ckpt_dir)
    if ck is not None:
        init_optim_state(opt)
        load_ckpt(model, opt, ck)
        if rank == 0:
            print(f"Resumed from {ck} at step {start}")
        for _ in range(start):
            sched.step()

    # data
    ds = SyntheticMultiView(a) if a.dataset == "synthetic" else RealMultiView(a)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(ds, batch_size=a.batch_per_gpu, sampler=sampler, shuffle=sampler is None,
                        num_workers=a.workers, pin_memory=True, drop_last=True,
                        persistent_workers=a.workers > 0)
    it, epoch = iter(loader), 0

    # constants for estimates
    n_tok = model.module.gh * model.module.gw if hasattr(model, "module") else 0
    tokens_step = a.batch_per_gpu * a.views * (n_tok + 1)
    flops_step = 6 * n_enc * tokens_step  # ~6*params*tokens; ignores attention and recompute
    f = (world - 1) / world
    comm_bytes_fsdp = 3 * f * n_params * 2          # 2 all-gathers + 1 reduce-scatter, bf16
    comm_bytes_ddp = 2 * f * n_params * 4           # ring all-reduce of fp32 grads
    mem_state_fsdp_gb = 16 * n_params / max(world, 1) / 1e9   # fp32 w + g + Adam(2) = 16 B/param
    mem_state_ddp_gb = 16 * n_params / 1e9

    micro = nccl_microbench(world) if start == 0 or True else {}
    sampler_gpu = GpuSampler(local)
    sampler_gpu.start()

    rows, win_step = {k: [] for k in ("data", "fwd", "bwd", "opt", "total", "mfu", "sps")}, []
    nccl_per_step, prof = None, None
    jsonl = open(out / "metrics.jsonl", "a") if rank == 0 else None
    t_seg = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    step = start

    while step < a.steps:
        if step == a.prof_start and prof is None:
            prof = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.start()

        t0 = time.perf_counter()
        try:
            views, target = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(loader)
            views, target = next(it)
        data_s = time.perf_counter() - t0
        views, target = views.cuda(non_blocking=True), target.cuda(non_blocking=True)

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        logits = model(views)
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), target,
            pos_weight=torch.tensor(a.pos_weight, device=logits.device))
        ev[1].record()
        loss.backward()
        ev[2].record()
        model.clip_grad_norm_(1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        ev[3].record()
        ev[3].synchronize()
        fwd, bwd, opt_s = (ev[0].elapsed_time(ev[1]) / 1e3, ev[1].elapsed_time(ev[2]) / 1e3,
                           ev[2].elapsed_time(ev[3]) / 1e3)
        total = time.perf_counter() - t0
        step += 1

        if prof is not None and step == a.prof_start + a.prof_steps:
            prof.stop()
            nccl_per_step = nccl_time_s(prof) / a.prof_steps
            prof = None

        if step > start + a.stats_skip:
            sps = a.batch_per_gpu * world / total
            mfu = flops_step / total / (a.peak_tflops * 1e12)
            for k, v in zip(rows, (data_s, fwd, bwd, opt_s, total, mfu, sps)):
                rows[k].append(v)
            win_step.append(total)

        if rank == 0:
            jsonl.write(json.dumps({
                "step": step, "loss": loss.item(), "data_s": data_s, "fwd_s": fwd, "bwd_s": bwd,
                "opt_s": opt_s, "step_s": total, "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
            }) + "\n")
            jsonl.flush()

        if step % a.log_every == 0 and win_step:
            t = torch.tensor([mean(win_step)], device="cuda")
            tmax, tmin = t.clone(), t.clone()
            if world > 1:
                dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
                dist.all_reduce(tmin, op=dist.ReduceOp.MIN)
            win_step.clear()
            if rank == 0:
                print(f"step {step} loss {loss.item():.4f} step_s {total:.3f} "
                      f"(straggler gap {tmax.item() - tmin.item():.3f}s) "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)

        # checkpoint / stop (all ranks must agree)
        stop = torch.tensor([1.0 if STOP_FLAG["stop"] else 0.0], device="cuda")
        if world > 1 and step % 10 == 0:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if step % a.ckpt_every == 0 or stop.item() > 0 or step == a.steps:
            save_ckpt(model, opt, step, a.ckpt_dir, rank)
        if stop.item() > 0:
            if rank == 0:
                print("SIGUSR1: checkpoint saved, exiting for requeue")
            break

    # ---- summary
    wall = time.perf_counter() - t_seg
    sampler_gpu.evt.set()
    ranks_max_mem = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device="cuda")
    if world > 1:
        dist.all_reduce(ranks_max_mem, op=dist.ReduceOp.MAX)
    if rank == 0:
        step_mean = mean(rows["total"])
        summary = {
            "run": a.run_name, "model": a.model, "params_M": n_params / 1e6, "world_size": world,
            "nodes": nnodes, "segment_steps": step - start, "segment_wall_s": wall,
            "node_hours_segment": wall / 3600 * nnodes,
            "gpu_hours_segment": wall / 3600 * world,
            "latency_s": {
                "step_mean": step_mean, "step_p50": pct(rows["total"], 0.5),
                "step_p95": pct(rows["total"], 0.95), "data": mean(rows["data"]),
                "forward": mean(rows["fwd"]), "backward": mean(rows["bwd"]),
                "optimizer": mean(rows["opt"]),
                "data_wait_fraction": mean(rows["data"]) / step_mean if step_mean else None},
            "throughput_samples_per_s": mean(rows["sps"]),
            "mfu": mean(rows["mfu"]),
            "memory_GB": {"peak_allocated_max_over_ranks": ranks_max_mem.item(),
                          "est_state_fsdp_per_gpu": mem_state_fsdp_gb,
                          "est_state_ddp_per_gpu": mem_state_ddp_gb},
            "communication": {
                "nccl_kernel_s_per_step_profiled": nccl_per_step,
                "est_volume_bytes_per_step_fsdp": comm_bytes_fsdp,
                "est_volume_bytes_per_step_ddp_equiv": comm_bytes_ddp,
                "effective_bw_GBps_vs_step_time": comm_bytes_fsdp / step_mean / 1e9 if step_mean else None,
                "effective_bw_GBps_vs_nccl_time": (comm_bytes_fsdp / nccl_per_step / 1e9
                                                   if nccl_per_step else None),
                "nccl_microbench": micro},
            "gpu_util_pct_mean": mean(sampler_gpu.u) if sampler_gpu.ok else None,
            "gpu_mem_ctrl_util_pct_mean": mean(sampler_gpu.m) if sampler_gpu.ok else None,
            "note": "NCCL kernel time includes waiting on peers and overlaps with compute; "
                    "it upper-bounds pure transfer time. FLOPs estimate is 6*params*tokens.",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        new = not Path(a.node_hours_csv).exists()
        with open(a.node_hours_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["user", "job_id", "run", "nodes", "wall_s", "node_hours", "steps"])
            w.writerow([getpass.getuser(), os.environ.get("SLURM_JOB_ID", ""), a.run_name, nnodes,
                        f"{wall:.0f}", f"{wall / 3600 * nnodes:.3f}", step - start])
        jsonl.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()