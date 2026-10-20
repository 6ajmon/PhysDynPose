#!/usr/bin/env python3
"""Install the official SMPL models where PhysDynPose (and FrankMocap) expect them.

The official pickles store ``shapedirs`` as a chumpy array, which is a problem on
numpy >= 1.24 (``from numpy import bool``). This script unpickles them with a
small numpy-alias shim, converts chumpy to plain numpy and writes the files:

    models/SMPL_male_10PCs.pkl                      <- SMPL v1.0.0 male (10 PCs)
    data/body_models/smpl/SMPL_MALE.pkl             <- SMPL v1.1.0 male
    data/body_models/smpl/SMPL_FEMALE.pkl           <- SMPL v1.1.0 female
    data/body_models/smpl/SMPL_NEUTRAL.pkl          <- SMPL v1.1.0 neutral
    data/J_regressor_extra.npy                      <- copied from SPIN / FrankMocap data
    <frankmocap>/extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

Usage:

    python tools/install_smpl_models.py \
        --v100 SMPL_python_v.1.0.0.zip \
        --v110 SMPL_python_v.1.1.0.zip \
        --frankmocap ~/frankmocap
"""

import argparse
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NBETAS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v100", type=Path, default=REPO / "SMPL_python_v.1.0.0.zip")
    parser.add_argument("--v110", type=Path, default=REPO / "SMPL_python_v.1.1.0.zip")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--frankmocap", type=Path, default=None,
                        help="Optional FrankMocap checkout; installs the neutral model there too")
    parser.add_argument("--keep-betas", action="store_true",
                        help=f"Keep all shape components instead of the first {NBETAS}")
    return parser.parse_args()


def import_chumpy():
    """Import chumpy on numpy >= 1.24 by restoring the removed aliases."""
    for name, value in [("bool", bool), ("int", int), ("float", float),
                        ("complex", complex), ("object", object), ("str", str)]:
        if not hasattr(np, name):
            setattr(np, name, value)
    if not hasattr(np, "unicode"):
        np.unicode = str
    try:
        import chumpy  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "chumpy is required to unpickle the official SMPL files "
            "(pip install --no-build-isolation chumpy==0.70)"
        ) from exc


def load_member(archive, suffix):
    """Load the first pickle in ``archive`` whose name ends with ``suffix``."""
    suffix = suffix.lower()
    with zipfile.ZipFile(archive) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".pkl") and n.lower().endswith(suffix)]
        if not members:
            raise SystemExit(f"{suffix} not found in {archive}")
        with zf.open(members[0]) as handle:
            data = pickle.load(handle, encoding="latin1")
    return data


def convert(data, nbetas):
    out = {}
    for key, value in data.items():
        if type(value).__module__.startswith("chumpy"):
            value = np.array(value)
        out[key] = value
    if nbetas is not None and out["shapedirs"].shape[2] > nbetas:
        out["shapedirs"] = np.ascontiguousarray(out["shapedirs"][:, :, :nbetas])
    return out


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=4)
    print(f"  wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, betas={data['shapedirs'].shape[2]})")


def verify(paths):
    import smplx

    for path in paths:
        model = smplx.SMPL(str(path), create_transl=False)
        import torch

        with torch.no_grad():
            out = model(betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 69),
                        global_orient=torch.zeros(1, 3))
        joint_count = out.joints.shape[1]
        pelvis = out.joints[0, 0].tolist()
        print(f"  verified {path.name}: joints={joint_count}, pelvis={[round(v, 3) for v in pelvis]}")


def main():
    args = parse_args()
    repo = args.repo.resolve()
    nbetas = None if args.keep_betas else NBETAS
    import_chumpy()

    print("SMPL v1.0.0 -> 10-PC male model")
    male10 = convert(load_member(args.v100, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"), None)
    save(male10, repo / "models" / "SMPL_male_10PCs.pkl")

    print("SMPL v1.1.0 -> body models")
    neutral = None
    targets = {"m": "SMPL_MALE.pkl", "f": "SMPL_FEMALE.pkl", "neutral": "SMPL_NEUTRAL.pkl"}
    out_paths = []
    for suffix, filename in targets.items():
        data = convert(load_member(args.v110, f"basicmodel_{suffix}_lbs_10_207_0_v1.1.0.pkl"), nbetas)
        path = repo / "data" / "body_models" / "smpl" / filename
        save(data, path)
        out_paths.append(path)
        if suffix == "neutral":
            neutral = data

    regressor = repo / "data" / "J_regressor_extra.npy"
    if args.frankmocap is not None:
        source = Path(args.frankmocap).expanduser() / "extra_data/body_module/data_from_spin/J_regressor_extra.npy"
        if source.exists():
            regressor.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, regressor)
            print(f"  copied {source} -> {regressor}")
        else:
            print(f"  [warn] {source} not found; J_regressor_extra.npy not installed", file=sys.stderr)

        fm_smpl = Path(args.frankmocap).expanduser() / "extra_data/smpl"
        if fm_smpl.is_dir() and neutral is not None:
            save(neutral, fm_smpl / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")

    print("Verifying with smplx")
    verify([repo / "models" / "SMPL_male_10PCs.pkl"] + out_paths)


if __name__ == "__main__":
    main()
