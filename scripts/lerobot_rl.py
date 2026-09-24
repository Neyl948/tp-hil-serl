"""Start a LeRobot RL process with the TP's compatibility fixes.

    python scripts/lerobot_rl.py learner --config_path runs/configs/<run>.json
    python scripts/lerobot_rl.py actor   --config_path runs/configs/<run>.json

LeRobot 0.6.1 types `ResetConfig.fixed_reset_joint_positions` as `Any | None`, and the
config serialiser (draccus) cannot encode that: the learner crashes at start-up with
"typing.Any cannot be used with isinstance()". We give the field its real type, then run
the normal LeRobot module unchanged. `train.py` prints the right command for you.

LeRobot also refuses to start a process whose output directory already exists, and the learner
creates it before the actor starts, so the actor gets its own sibling directory (`<dir>_actor`).

The processes that open the simulator (teleoperation, recording, the actor) also get the TP's
keyboard controls, set up by `teleop.install()` (see `teleop.py`).
"""

import dataclasses
import json
import runpy
import sys

MODULES = {"learner": "lerobot.rl.learner", "actor": "lerobot.rl.actor",
           "gym_manipulator": "lerobot.rl.gym_manipulator"}


def patch_reset_config():
    from lerobot.envs.configs import ResetConfig

    name = "fixed_reset_joint_positions"
    ResetConfig.__annotations__[name] = list[float] | None
    for f in dataclasses.fields(ResetConfig):
        if f.name == name:
            f.type = list[float] | None


def load_config(argv):
    """The JSON config passed with --config_path, or {}."""
    for i, a in enumerate(argv):
        path = argv[i + 1] if a == "--config_path" and i + 1 < len(argv) else (
            a.split("=", 1)[1] if a.startswith("--config_path=") else None)
        if path:
            return json.load(open(path))
    return {}


def actor_output_args(argv, cfg):
    """`--output_dir=<run dir>_actor` for the actor, unless the caller already set one."""
    if any(a.startswith("--output_dir") for a in argv):
        return []
    out = cfg.get("output_dir")
    return [f"--output_dir={out}_actor"] if out else []


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        sys.exit(f"usage: python scripts/lerobot_rl.py {{{'|'.join(MODULES)}}} --config_path <file>")
    module = MODULES[sys.argv[1]]
    patch_reset_config()
    cfg = load_config(sys.argv[2:])
    if sys.argv[1] in ("actor", "gym_manipulator"):
        import teleop
        # Without a policy (teleop, record) the operator ends each episode with Enter.
        teleop.install(cfg.get("env", {}).get("fps"), success_by_key=sys.argv[1] == "gym_manipulator")
    extra = actor_output_args(sys.argv[2:], cfg) if sys.argv[1] == "actor" else []
    sys.argv = [module] + sys.argv[2:] + extra
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
