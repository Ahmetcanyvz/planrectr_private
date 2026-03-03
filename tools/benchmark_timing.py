#!/usr/bin/env python3
"""
Benchmark PlaneRecTR per-frame inference timing.

Runs on 10 random ScanNet++ test scenes (100 frames each, 1000 total) and
reports mean ms and FPS.

Usage:
    python tools/benchmark_timing.py
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor

from PlaneRecTR import add_PlaneRecTR_config


def setup_cfg(config_file, checkpoint, device):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_PlaneRecTR_config(cfg)
    cfg.merge_from_file(config_file)
    if checkpoint:
        cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def load_scannetpp_frames(rgb_root, gt_root, scene_id, n_frames=100, out_h=480, out_w=640):
    """Load n_frames from a ScanNet++ scene."""
    gt_h5 = os.path.join(gt_root, scene_id, "rendered.h5")
    with h5py.File(gt_h5, "r") as hf:
        all_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                    for fid in hf["frame_ids"][:]]
    rgbs = []
    for fid in all_fids:
        if len(rgbs) >= n_frames:
            break
        rgb_path = os.path.join(rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
        if not os.path.exists(rgb_path):
            continue
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        rgbs.append(cv2.resize(rgb, (out_w, out_h)))
    return rgbs


def load_multi_scene_frames(rgb_root, gt_root, splits_root, n_scenes=10, frames_per_scene=100):
    """Load frames from multiple random ScanNet++ test scenes."""
    import random
    random.seed(42)
    split_file = os.path.join(splits_root, "scannetpp", "nvs_sem_test_with_planes.txt")
    with open(split_file) as f:
        all_scenes = [l.strip() for l in f if l.strip()]

    valid_scenes = []
    for sid in all_scenes:
        gt_h5 = os.path.join(gt_root, sid, "rendered.h5")
        rgb_dir = os.path.join(rgb_root, sid, "iphone", "rgb")
        if os.path.exists(gt_h5) and os.path.isdir(rgb_dir):
            valid_scenes.append(sid)

    n_scenes = min(n_scenes, len(valid_scenes))
    selected = random.sample(valid_scenes, n_scenes)
    selected.sort()

    all_rgbs = []
    for sid in selected:
        rgbs = load_scannetpp_frames(rgb_root, gt_root, sid, frames_per_scene)
        print(f"  {sid}: {len(rgbs)} frames")
        all_rgbs.extend(rgbs)

    return all_rgbs, selected


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark PlaneRecTR inference timing")
    p.add_argument("--config_file", type=str,
                   default="configs/PlaneRecTRScanNetV1/PlaneRecTR_R50_demo.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_scenes", type=int, default=10)
    p.add_argument("--frames_per_scene", type=int, default=100)
    p.add_argument("--scannetpp_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")

    planamono_root = str(Path(__file__).resolve().parents[1].parent / "PixelwisePlanarity" / "planamono")
    p.add_argument("--splits_root", type=str,
                   default=os.path.join(planamono_root, "splits"))
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Scenes: {args.n_scenes}")
    print(f"Frames per scene: {args.frames_per_scene}")

    print("Loading frames...")
    rgbs, scene_ids = load_multi_scene_frames(
        args.scannetpp_rgb_root, args.scannetpp_gt_root,
        args.splits_root, args.n_scenes, args.frames_per_scene)
    print(f"Loaded {len(rgbs)} frames from {len(scene_ids)} scenes")

    cfg = setup_cfg(args.config_file, args.checkpoint, args.device)
    predictor = DefaultPredictor(cfg)
    infer_h, infer_w = cfg.INPUT.IMAGE_SIZE

    print(f"Config: {args.config_file}")
    print(f"Checkpoint: {cfg.MODEL.WEIGHTS}")
    print(f"Infer res: {infer_h}x{infer_w}")

    times = []
    for rgb in tqdm(rgbs, desc="PlaneRecTR"):
        img_resized = cv2.resize(rgb, (infer_w, infer_h))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            predictions = predictor(img_bgr)
        sem_seg = predictions["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.uint16)
        if sem_seg.shape[0] != 480 or sem_seg.shape[1] != 640:
            sem_seg = cv2.resize(sem_seg, (640, 480), interpolation=cv2.INTER_NEAREST)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    # Skip first 5 frames (GPU warmup)
    times = times[5:]
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / mean_ms

    print(f"\n{'=' * 60}")
    print(f"  PlaneRecTR Timing ({len(times)} frames)")
    print(f"{'=' * 60}")
    print(f"  Mean: {mean_ms:.1f} ms  (std {std_ms:.1f})")
    print(f"  FPS:  {fps:.1f}")


if __name__ == "__main__":
    main()
