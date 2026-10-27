"""Rescale 4D-Humans root translations to the real camera focal length.

The stock 4D-Humans ``demo.py`` converts the weak-perspective camera prediction to a
3D root translation with a hard-coded focal length
(``cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(image_size)``), not with the
intrinsics of the recording it runs on.  For MoviCam that focal length is ~30x too
large, so ``pred_cam_t_full`` comes out ~30x too deep (~70 m instead of ~2 m) and the
physics optimization becomes infeasible.

In ``cam_crop_to_full`` only the depth component depends on the focal length, so the
fix is a single rescale of ``z`` by ``f_true / f_assumed``.

Usage:
    python tools/fix_out_list_focal.py --seq_no 6 --surface 31
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import sequence_frame_range
from release_paths import camera_exts_txt, fourdhumans_out_list

# Defaults of the 4D-Humans demo conversion (hmr2/configs/__init__.py).
ASSUMED_FOCAL_LENGTH = 5000.0
ASSUMED_IMAGE_SIZE = 224.0


def first_intrinsic_fx(seq_no: int, surface: int) -> float:
    start_frame, _ = sequence_frame_range(seq_no, surface)
    lines = camera_exts_txt(seq_no, surface).read_text().strip().splitlines()
    intrinsics = np.array([[float(x) for x in line.split()[1:]] for line in lines[1::3]]).reshape(-1, 4, 4)
    return float(intrinsics[start_frame, 0, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_no", type=int, required=True)
    parser.add_argument("--surface", type=int, required=True)
    parser.add_argument(
        "--assumed_focal",
        type=float,
        default=None,
        help="Focal length (px, full image) used when out_list.pkl was produced",
    )
    parser.add_argument(
        "--image_max",
        type=float,
        default=1920.0,
        help="Longest image side of the frames 4D-Humans ran on",
    )
    args = parser.parse_args()

    trajectory = fourdhumans_out_list(args.seq_no, args.surface)
    if not trajectory.exists():
        raise FileNotFoundError(f"Missing 4D-Humans output: {trajectory}")

    focal_true = first_intrinsic_fx(args.seq_no, args.surface)
    if args.assumed_focal is not None:
        focal_assumed = args.assumed_focal
    else:
        focal_assumed = ASSUMED_FOCAL_LENGTH / ASSUMED_IMAGE_SIZE * args.image_max
    if focal_assumed <= 0 or focal_true <= 0:
        raise ValueError("Focal lengths must be positive")

    scale = focal_true / focal_assumed
    tracks = joblib.load(trajectory)
    for track in tracks:
        translation = np.asarray(track["pred_cam_t_full"], dtype=np.float64).reshape(-1)
        translation[2] *= scale
        track["pred_cam_t_full"] = translation.reshape(1, 3).astype(np.float32)
    joblib.dump(tracks, trajectory)

    print(
        f"Rescaled {len(tracks)} tracks in {trajectory}: "
        f"f_true={focal_true:.2f} f_assumed={focal_assumed:.2f} scale={scale:.6f}"
    )


if __name__ == "__main__":
    main()
