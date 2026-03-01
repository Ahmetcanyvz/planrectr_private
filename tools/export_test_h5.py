#!/usr/bin/env python3
"""
Run PlaneRecTR inference on test sets and save plane segmentation to H5.

Supports 4 datasets: scannetpp, hypersim, vkitti2, synthia.
Output format matches PixelwisePlanarity conventions: per-scene H5 with
frame_ids (string) and planes (uint16) datasets.

Usage:
    python tools/export_test_h5.py --dataset scannetpp
    python tools/export_test_h5.py --dataset all
    python tools/export_test_h5.py --dataset vkitti2 --max_frames 10
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
import cv2
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor

from PlaneRecTR import add_PlaneRecTR_config


# ============================================================
# Model setup
# ============================================================

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


def run_planrectr(predictor, rgb_uint8, out_h, out_w, infer_h, infer_w):
    """
    Run PlaneRecTR on a single RGB image.

    Args:
        predictor: detectron2 DefaultPredictor
        rgb_uint8: (H, W, 3) uint8 RGB image
        out_h, out_w: output resolution for the plane segmentation
        infer_h, infer_w: model inference resolution (from config IMAGE_SIZE)

    Returns:
        labels: (out_h, out_w) uint16 plane segmentation (0=non-planar, 1..K=planes)
    """
    img_resized = cv2.resize(rgb_uint8, (infer_w, infer_h))

    # DefaultPredictor expects BGR
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        predictions = predictor(img_bgr)

    # sem_seg: (num_queries+1, 192, 256) — argmax gives per-pixel plane ID
    sem_seg = predictions["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.uint16)

    # Resize to output resolution
    if sem_seg.shape[0] != out_h or sem_seg.shape[1] != out_w:
        sem_seg = cv2.resize(sem_seg, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    return sem_seg.astype(np.uint16)


# ============================================================
# Hypersim HDR loader
# ============================================================

def load_hypersim_hdr_rgb(h5_path, percentile=90, target_max=0.8, gamma=2.2):
    with h5py.File(h5_path, 'r') as f:
        hdr = f['dataset'][:].astype(np.float32)
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=1e4, neginf=0.0)
    hdr = np.clip(hdr, 0, 1e4)
    brightness = hdr.mean(axis=2)
    scale_val = np.nanpercentile(brightness, percentile)
    scale_val = max(scale_val, 1e-6) if np.isfinite(scale_val) else 1.0
    img = hdr * (target_max / scale_val)
    img = np.clip(img, 0, None) ** (1.0 / gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ============================================================
# Scene-level iterators: yield (scene_label, h5_relpath, frame_ids, rgbs)
# ============================================================

def iter_scannetpp_scenes(args):
    split_file = os.path.join(args.splits_root, "scannetpp", "nvs_sem_test_with_planes.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene_id in scenes:
        gt_h5 = os.path.join(args.scannetpp_gt_root, scene_id, "rendered.h5")
        if not os.path.exists(gt_h5):
            continue
        with h5py.File(gt_h5, "r") as hf:
            all_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                        for fid in hf["frame_ids"][:]]
        frame_ids, rgbs = [], []
        for fid in all_fids:
            rgb_path = os.path.join(args.scannetpp_rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
            if not os.path.exists(rgb_path):
                continue
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rgbs.append(cv2.resize(rgb, (args.width, args.height)))
            frame_ids.append(fid)
        if frame_ids:
            yield scene_id, os.path.join(scene_id, "rendered_v2.h5"), frame_ids, rgbs


def iter_hypersim_scenes(args):
    split_csv = os.path.join(args.splits_root, "hypersim",
                             "metadata_images_split_with_planes_filtered.csv")
    df = pd.read_csv(split_csv)
    test_df = df[df["split_partition_name"] == "test"]
    groups = test_df.groupby(["scene_name", "camera_name"])
    for (scene, cam), group_df in groups:
        frame_ids, rgbs = [], []
        for fid in sorted(group_df["frame_id"].tolist()):
            fid = int(fid)
            rgb_path = os.path.join(
                args.hypersim_data_root, scene, "images",
                f"scene_{cam}_final_hdf5", f"frame.{fid:04d}.color.hdf5")
            if not os.path.exists(rgb_path):
                continue
            rgb = load_hypersim_hdr_rgb(rgb_path)
            rgbs.append(cv2.resize(rgb, (args.width, args.height)))
            frame_ids.append(f"{fid:04d}")
        if frame_ids:
            yield f"{scene}/{cam}", os.path.join(scene, f"rendered_planes_{cam}.h5"), frame_ids, rgbs


def iter_vkitti2_scenes(args):
    split_file = os.path.join(args.splits_root, "vkitti2", "test.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene in scenes:
        h5_path = os.path.join(args.vkitti2_plane_root, scene, "clone", "scene_data.h5")
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, "r") as hf:
            n = hf["rgb"].shape[0]
            frame_ids = [f"{i:04d}" for i in range(n)]
            rgbs = [cv2.resize(hf["rgb"][i], (args.width, args.height)) for i in range(n)]
        yield f"{scene}/clone", os.path.join(scene, "clone", "rendered_v2.h5"), frame_ids, rgbs


def iter_synthia_scenes(args):
    split_file = os.path.join(args.splits_root, "synthia", "test.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene in scenes:
        h5_path = os.path.join(args.synthia_plane_root, "test", scene, "scene_data.h5")
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, "r") as hf:
            n = hf["rgb"].shape[0]
            frame_ids = [f"{i:04d}" for i in range(n)]
            rgbs = [cv2.resize(hf["rgb"][i], (args.width, args.height)) for i in range(n)]
        yield scene, os.path.join(scene, "rendered_v2.h5"), frame_ids, rgbs


DATASET_ITERS = {
    "scannetpp": iter_scannetpp_scenes,
    "hypersim": iter_hypersim_scenes,
    "vkitti2": iter_vkitti2_scenes,
    "synthia": iter_synthia_scenes,
}


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Export PlaneRecTR inference to H5")
    p.add_argument("--config_file", type=str,
                   default="configs/PlaneRecTRScanNetV1/PlaneRecTR_R50_demo.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override MODEL.WEIGHTS (default: from config)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output root (default: /cluster/scratch/ayavuz/dataset/planrectr_{dataset})")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["scannetpp", "hypersim", "vkitti2", "synthia", "all"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)

    # Dataset paths
    planamono_root = str(Path(__file__).resolve().parents[1].parent / "PixelwisePlanarity" / "planamono")
    p.add_argument("--splits_root", type=str,
                   default=os.path.join(planamono_root, "splits"))
    p.add_argument("--scannetpp_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--hypersim_data_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/hypersim")
    p.add_argument("--vkitti2_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/vkitti2_planes")
    p.add_argument("--synthia_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/synthia_planes")
    return p.parse_args()


def export_dataset(dataset_name, predictor, args, infer_h, infer_w):
    if args.output_dir:
        ds_out = os.path.join(args.output_dir, dataset_name)
    else:
        ds_out = f"/cluster/scratch/ayavuz/dataset/planrectr_{dataset_name}"
    os.makedirs(ds_out, exist_ok=True)

    total_frames = 0
    total_scenes = 0

    for scene_label, h5_rel, frame_ids, rgbs in DATASET_ITERS[dataset_name](args):
        if args.max_frames is not None and total_frames >= args.max_frames:
            break

        n = len(frame_ids)
        if args.max_frames is not None:
            n = min(n, args.max_frames - total_frames)
            frame_ids = frame_ids[:n]
            rgbs = rgbs[:n]

        planes_all = np.zeros((n, args.height, args.width), dtype=np.uint16)

        for i, rgb in enumerate(tqdm(rgbs, desc=f"  {scene_label}", leave=False)):
            planes_all[i] = run_planrectr(predictor, rgb, args.height, args.width, infer_h, infer_w)

        out_h5 = os.path.join(ds_out, h5_rel)
        os.makedirs(os.path.dirname(out_h5), exist_ok=True)
        with h5py.File(out_h5, "w") as f:
            dt = h5py.string_dtype()
            f.create_dataset("frame_ids", data=frame_ids, dtype=dt)
            f.create_dataset("planes", data=planes_all, dtype=np.uint16)

        total_frames += n
        total_scenes += 1
        tqdm.write(f"  Saved: {out_h5} ({n} frames)")

    print(f"  {dataset_name}: {total_scenes} scene(s), {total_frames} frames -> {ds_out}")


def main():
    args = parse_args()
    datasets = list(DATASET_ITERS.keys()) if args.dataset == "all" else [args.dataset]

    cfg = setup_cfg(args.config_file, args.checkpoint, args.device)
    predictor = DefaultPredictor(cfg)

    # Read inference resolution from config
    infer_h, infer_w = cfg.INPUT.IMAGE_SIZE
    out_label = args.output_dir or "/cluster/scratch/ayavuz/dataset/planrectr_{dataset}"

    print("PlaneRecTR Export")
    print("=" * 60)
    print(f"Config:     {args.config_file}")
    print(f"Checkpoint: {cfg.MODEL.WEIGHTS}")
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Output:     {out_label}")
    print(f"Infer res:  {infer_h}x{infer_w}")
    print(f"Output res: {args.height}x{args.width}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames} per dataset")
    print("=" * 60)

    for ds in datasets:
        print(f"\n--- {ds} ---")
        export_dataset(ds, predictor, args, infer_h, infer_w)

    print("\nDone!")


if __name__ == "__main__":
    main()
