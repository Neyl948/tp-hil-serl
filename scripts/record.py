"""Part 2: record your own demonstrations, then print a summary of what was recorded.

    python scripts/record.py                    # 10 episodes
    python scripts/record.py --episodes 5
    python scripts/record.py --summary-only     # just inspect an existing recording

Each episode: press Space to take control, grasp and lift the cube (hold C), then press SUCCESS (V or Enter).
Messed up? RE-RECORD (X) or FAILURE (Esc).
The dataset is saved locally in runs/data/pick_cube_<group>/ (never pushed to the Hub).
"""

import argparse
import shutil

from teleop import TELEOP_EPISODE_S
from common import (add_common_args, check_display, demos_repo, demos_root,
                    detect_device, load_reference, make_config, require_group, run_module)


def summary(group):
    root = demos_root(group)
    if not root.exists():
        print(f"No dataset found in {root}")
        return
    try:
        import numpy as np
        import pandas as pd
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(demos_repo(group), root=root)
        print(f"\nDataset {demos_repo(group)}: {ds.num_episodes} episodes, {ds.num_frames} frames at {ds.fps} fps")
        reward_key = next((k for k in ds.features if "reward" in k), None)
        df = pd.DataFrame({"episode": np.asarray(ds.hf_dataset["episode_index"])})
        if reward_key:
            df["reward"] = np.asarray(ds.hf_dataset[reward_key])
            per_ep = df.groupby("episode").agg(length=("reward", "size"), final_reward=("reward", "max"))
        else:
            per_ep = df.groupby("episode").size().rename("length").to_frame()
        per_ep["duration_s"] = per_ep["length"] / ds.fps
        print(per_ep.to_string())
        print("\nFeatures:", ", ".join(ds.features))
    except Exception as e:  # noqa: BLE001  (API differences between LeRobot versions)
        print(f"Could not summarise the dataset automatically ({type(e).__name__}: {e}).")
        print(f"The raw files are in {root}: look at meta/info.json and meta/episodes*.")


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Delete a previous recording first")
    args = p.parse_args()
    group = require_group(args)

    if args.summary_only:
        summary(group)
        return

    check_display()
    root = demos_root(group)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} already exists. Use --overwrite to start again, or --summary-only to inspect it.")
        shutil.rmtree(root)

    base = load_reference("env_config.json")
    cfg = make_config(base, {
        "mode": "record",
        "device": detect_device(),
        "dataset.repo_id": demos_repo(group),
        "dataset.root": str(root),
        "dataset.task": base["env"]["task"],
        "dataset.num_episodes_to_record": args.episodes,
        "dataset.replay_episode": None,
        "dataset.push_to_hub": False,
        "env.processor.reset.control_time_s": TELEOP_EPISODE_S,
    }, f"{group}_record")
    run_module("gym_manipulator", cfg)
    summary(group)


if __name__ == "__main__":
    main()
