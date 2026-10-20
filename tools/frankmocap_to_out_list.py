#!/usr/bin/env python3
"""Run FrankMocap's body module on a frame folder and write the ``out_list.pkl``
that ``align_slam.py`` expects.

Output format (one entry per frame, in file-name order):

    out_list[i] = {
        "pred_smpl_params": {
            "betas": (1, 10),
            "body_pose": (1, 23, 3, 3),   # rotation matrices
            "global_orient": (1, 3, 3),
        },
        "pred_cam_t_full": (1, 3),         # root translation in camera space [m]
    }

FrankMocap regresses SMPL in a weak-perspective crop, the same convention as
SPIN/HMR. ``cam_crop_to_full`` below is the standard weak-perspective to
full-perspective conversion (same formula as in 4D-Humans), so the result is
directly comparable with what HMR2 produced for PhysDynPose.

Run it with the FrankMocap environment, e.g.:

    python tools/frankmocap_to_out_list.py \
        --mocap_root ~/frankmocap \
        --frames_dir outputs/raw_frames/seq1_31 \
        --out_dir inputs/4d_humans/outputs_01 \
        --focal_px 1000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

DEFAULT_CHECKPOINT = "extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt"
DEFAULT_SMPL_DIR = "extra_data/smpl"
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocap_root", type=Path, required=True, help="FrankMocap checkout")
    parser.add_argument("--frames_dir", type=Path, required=True, help="Raw frames, one person per frame")
    parser.add_argument("--out_dir", type=Path, required=True, help="Where out_list.pkl is written")
    parser.add_argument("--smpl_dir", type=Path, default=None, help="Default: <mocap_root>/extra_data/smpl")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Default: <mocap_root>/" + DEFAULT_CHECKPOINT)
    parser.add_argument("--focal_px", type=float, default=None,
                        help="Focal length in pixels. Default: 5000/256*max(H,W), as in 4D-Humans")
    parser.add_argument("--intrinsics", type=Path, default=None,
                        help="camera_exts_full.txt; reads fx from --frame_index (overrides --focal_px)")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Sequence frame index inside --intrinsics (the first frame of the range)")
    parser.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                        help="Use a fixed bbox instead of running the 2D detector")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def cam_crop_to_full(cam, box_center, box_size, img_size, focal_length):
    """Weak-perspective crop camera -> full-perspective camera-space translation.

    Mirrors ``hmr2.utils.renderer.cam_crop_to_full`` (4D-Humans).
    """
    img_w, img_h = img_size
    s, tx, ty = cam
    bs = float(box_size) * float(s) + 1e-9
    tz = 2.0 * focal_length / bs
    tx_full = 2.0 * (box_center[0] - img_w / 2.0) / bs + tx
    ty_full = 2.0 * (box_center[1] - img_h / 2.0) / bs + ty
    return np.array([tx_full, ty_full, tz], dtype=np.float32)


def list_frames(frames_dir, max_frames):
    frames = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    return frames[:max_frames] if max_frames else frames


def main():
    args = parse_args()
    mocap_root = args.mocap_root.resolve()
    frames_dir = args.frames_dir.resolve()
    out_dir = args.out_dir.resolve()
    smpl_dir = (args.smpl_dir or mocap_root / DEFAULT_SMPL_DIR).resolve()
    checkpoint = (args.checkpoint or mocap_root / DEFAULT_CHECKPOINT).resolve()

    if not (mocap_root / "bodymocap").is_dir():
        raise SystemExit(f"Not a FrankMocap checkout: {mocap_root}")
    if not checkpoint.exists():
        raise SystemExit(f"Missing body module checkpoint: {checkpoint}")

    # FrankMocap resolves extra_data/* relative to the current directory.
    os.chdir(mocap_root)
    sys.path.insert(0, str(mocap_root))

    import torch
    from bodymocap.body_bbox_detector import BodyPoseEstimator
    from bodymocap.body_mocap_api import BodyMocap
    from bodymocap import constants

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    frames = list_frames(frames_dir, args.max_frames)
    if not frames:
        raise SystemExit(f"No frames found in {frames_dir}")

    detector = None if args.bbox is not None else BodyPoseEstimator()
    body = BodyMocap(str(checkpoint), str(smpl_dir), device=device)
    out_dir.mkdir(parents=True, exist_ok=True)

    focal_override = args.focal_px
    if args.intrinsics is not None:
        lines = Path(args.intrinsics).read_text().splitlines()
        intrinsic_rows = lines[1::3]
        if args.frame_index >= len(intrinsic_rows):
            raise SystemExit(f"--frame_index {args.frame_index} out of range for {args.intrinsics}")
        focal_override = float(intrinsic_rows[args.frame_index].split()[1])

    outputs = []
    last_entry = None
    missing = 0
    start = time.time()

    for index, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise SystemExit(f"Cannot read frame: {frame_path}")

        if args.bbox is not None:
            boxes = [np.asarray(args.bbox, dtype=np.float64)]
        else:
            _, boxes = detector.detect_body_pose(frame)
            boxes = list(boxes)

        if not boxes:
            missing += 1
            if last_entry is None:
                print(f"[warn] no person in first frame {frame_path.name}; skipping", flush=True)
                continue
            outputs.append(last_entry)
            continue

        # single-subject pipeline: keep the largest detection
        box = max(boxes, key=lambda b: float(b[2]) * float(b[3]))
        pred = body.regress(frame, [box])[0]

        rotmat = pred["pred_rotmat"].astype(np.float32)          # (1, 24, 3, 3)
        betas = pred["pred_betas"].astype(np.float32)            # (1, 10)

        height, width = frame.shape[:2]
        focal = focal_override if focal_override else 5000.0 / 256.0 * max(width, height)
        scale_ratio = float(pred["bbox_scale_ratio"])
        half = 0.5 * constants.IMG_RES / scale_ratio
        box_size = constants.IMG_RES / scale_ratio
        center = (float(pred["bbox_top_left"][0]) + half, float(pred["bbox_top_left"][1]) + half)
        cam_t = cam_crop_to_full(pred["pred_camera"], center, box_size, (width, height), focal)

        entry = {
            "pred_smpl_params": {
                "betas": betas,
                "body_pose": rotmat[:, 1:],
                "global_orient": rotmat[:, 0],
            },
            "pred_cam_t_full": cam_t[None],
        }
        outputs.append(entry)
        last_entry = entry

        if index % 10 == 0 or index == len(frames) - 1:
            elapsed = time.time() - start
            print(f"[{index + 1}/{len(frames)}] {elapsed / (index + 1):.2f} s/frame", flush=True)

    if not outputs:
        raise SystemExit("FrankMocap produced no predictions")

    if len(outputs) != len(frames):
        raise SystemExit(f"Expected {len(frames)} predictions, got {len(outputs)}")

    output_path = out_dir / "out_list.pkl"
    joblib.dump(outputs, output_path)

    elapsed = time.time() - start
    print(f"Saved {len(outputs)} frames to {output_path} "
          f"({elapsed:.1f} s, {len(outputs) / elapsed:.2f} fps, {missing} frames without detection)")


if __name__ == "__main__":
    main()
