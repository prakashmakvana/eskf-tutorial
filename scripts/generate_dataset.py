"""Generate the synthetic IMU + 6D pose dataset into ./data

    python scripts/generate_dataset.py --seed 0 --duration 60
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eskf import SimConfig, simulate, save_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--duration", type=float, default=60.0)
args = ap.parse_args()

truth, imu, pose = simulate(SimConfig(seed=args.seed, duration=args.duration))
save_dataset(args.out, truth, imu, pose)
print(f"wrote {len(imu)} IMU samples and {len(pose)} pose measurements to {os.path.abspath(args.out)}")
