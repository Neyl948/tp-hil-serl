# TP: Interactive Robot Learning with HIL-SERL

A 4-hour lab on **human-in-the-loop reinforcement learning**. Students train a simulated Franka Panda arm to pick up a cube with SAC, first on its own, then while they coach it by taking over with the keyboard, and measure how much their interventions speed up learning.

Built on [LeRobot](https://github.com/huggingface/lerobot)'s HIL-SERL implementation and the [`gym_hil`](https://github.com/huggingface/gym-hil) MuJoCo environment. No real robot needed.

**Students: the lab handout with all the exercises is [`TP.md`](TP.md). Write your report in [`answers.md`](answers.md).**

> **Submission deadline: Thursday 1 October 2026, before midnight** (one week after the TP). One report per group: list **all group members** (full name and GitHub username of each) at the top of `answers.md`, and each member registers in the class spreadsheet. How to hand in: [Hand-in on GitHub](TP.md#hand-in-on-github).

| Part | Duration | Content | Script |
| --- | --- | --- | --- |
| Setup | 15 min | Check the machine, controls, W&B | `check_setup.py` |
| 1 | 30 min | Explore the environment, teleoperate | `inspect_env.py`, `teleop.py` |
| 2 | 20 min | Record 10 demonstrations | `record.py` |
| 3 | 35 min | RL **without** interventions (baseline) | `train.py --run noHIL` |
| 4 | 40 min | HIL-SERL **with** interventions | `train.py --run HIL`, `plot_results.py` |
| 5 | 45 min | One controlled experiment per group | `train.py --run exp --exp X` |
| 6 | 30 min | Class comparison and report | `plot_results.py --class` |

## Requirements

Everything runs **on the machine the student is sitting at**: the simulator opens a 3D window and reads the local keyboard. Linux, macOS and Windows are all supported; `check_setup.py` auto-detects the best available compute device (CUDA, Apple MPS, or CPU) and configures every run for it, so the same commands work everywhere.

- **Linux**: any desktop session using **X11** (not Wayland: keyboard input is ignored under Wayland). An NVIDIA GPU (`nvidia-smi` works) gives the fastest training; without one, training falls back to CPU and is much slower but still works.
- **macOS**: Apple Silicon uses the GPU via MPS automatically; Intel Macs fall back to CPU. The first time you teleoperate, macOS may ask you to grant your terminal app Accessibility / Input Monitoring permission (System Settings > Privacy & Security) so it can capture the keyboard.
- **Windows**: two options —
  - **WSL2 + NVIDIA GPU** (fastest, closest to a Linux lab machine): install a recent NVIDIA driver on the *Windows* side, `wsl --install` (Windows 11 includes WSLg, so the simulator window appears natively, no extra X server needed), then run everything from `install.sh` onward *inside* the WSL2 Ubuntu shell.
  - **Native Windows, CPU-only**: use `install.ps1` from an Anaconda Prompt / PowerShell. Works on any Windows laptop, no WSL required, but training is CPU-speed.
- [Miniconda](https://docs.anaconda.com/miniconda/) and git, on whichever OS/shell you install into.
- A free [Weights & Biases](https://wandb.ai) account per student (recommended; the TP also works offline with the observer logs)

**On CPU or MPS, training is slower than on the GPU lab machines the schedule was timed against** — Parts 3-5 may need more than the suggested duration to show the same learning curves. That's expected, not a bug: note your device (`cuda` / `mps` / `cpu`, and GPU model if any) in `answers.md`, since it matters for comparing results in Part 6.

It does **not** work on Google Colab, a remote JupyterHub or over plain SSH: there is no screen for the simulator and your keyboard is not the server's. A remote desktop (NoMachine, VNC, X2Go) into a GPU machine does work, and so does WSL2 (which behaves like a local desktop session via WSLg).

## Install (once per machine)

```bash
git clone https://github.com/Silviatulli/tp-hil-serl.git
cd tp-hil-serl
bash install.sh          # Linux, macOS, or Windows-via-WSL2: conda env "tp-hil" with LeRobot 0.6.1 + HIL-SERL extras
conda activate tp-hil
python prefetch.py       # downloads the 30 reference demos + vision encoder
python check_setup.py    # every line should be PASS or WARN, no FAIL
```

On native Windows (no WSL2), use `install.ps1` instead of `install.sh` from an Anaconda Prompt / PowerShell; `prefetch.py` and `check_setup.py` are the same on every OS.

Clone the course repository; don't fork it. A fork of a public repository is public, so your answers would be visible to everyone. You hand in through your own private repository instead (see [Hand-in on GitHub](TP.md#hand-in-on-github)).

## Getting updates

The course repository may be updated during the TP (fixes, changed commands). To get the latest version, from the repository root:

```bash
git remote add upstream https://github.com/Silviatulli/tp-hil-serl.git   # once
git add -A && git commit -m "my work so far"                             # save your changes first (skip if nothing to commit)
git pull --no-rebase upstream main                                         # merge the update into your copy
```

This works whether you cloned the course repository, forked it, or already pointed `origin` to your own repository for the hand-in. Your runs in `runs/` are never touched. No reinstall is needed unless the update says so.

## Quick start

In a terminal **opened inside the desktop session**, from the repository root (native Windows: use an Anaconda Prompt / PowerShell and `$env:TP_GROUP="group07"` instead of `export`):

```bash
conda activate tp-hil
export TP_GROUP=group07             # your group name, in every terminal
python scripts/inspect_env.py       # Part 1.1
python scripts/teleop.py            # Part 1.2
python scripts/record.py            # Part 2
python scripts/train.py --run noHIL # Part 3: prints the learner/actor commands to run
```

Then follow [`TP.md`](TP.md). Everything you produce goes to `runs/` (ignored by git).

**Hand-in (deadline Thursday 1 October 2026, before midnight):** students push their report to their own private GitHub repository, add the instructor (`Silviatulli`) as a collaborator, and register their name, GitHub username and repository link in the [class spreadsheet](https://docs.google.com/spreadsheets/d/1z7GuBLEifa7PDZplGhnKOQXfGCPMJKUpL1sx0CHchBA/edit?usp=sharing). Details in [`TP.md`](TP.md#hand-in-on-github).

## Repository layout

```
tp-hil-serl/
├── TP.md                  lab handout: background, instructions, exercises
├── answers.md             report template to fill in
├── configs/               reference gym_hil configs from LeRobot (never modified)
├── scripts/
│   ├── inspect_env.py     Part 1.1: spaces, camera views, state ranges
│   ├── teleop.py          Part 1.2: drive the robot
│   ├── record.py          Part 2: record demonstrations + summary
│   ├── train.py           Parts 3-5: write a run config, print the learner/actor commands
│   ├── log_progress.py    observer log (first success, successes/10, interventions)
│   ├── plot_results.py    Parts 4-6: learning curves, group and class comparison
│   └── common.py          shared paths and config helpers
├── install.sh             creates the conda env (Linux, macOS, Windows-via-WSL2)
├── install.ps1            creates the conda env on native Windows (CPU-only)
├── prefetch.py            caches the dataset and encoder from the Hugging Face Hub
├── check_setup.py         machine check: device (CUDA/MPS/CPU), display, rendering, keyboard, cache, W&B
└── instructor/            preparation checklist, pitfalls, grading, answer key
```

## Controls

| Action | Key |
| --- | --- |
| Move in x–y plane | Arrow keys |
| Move up / down (z) | U / D (tap for one step, hold to keep moving) |
| Grasp | Hold C (the gripper opens when you release it) |
| **Take over from the policy** | **Space** (toggle) |
| End episode: success | V (next to C, so you can keep holding C) or Enter |
| End episode: failure | Esc |
| Re-record episode | X |

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `GLFWError: X11: The DISPLAY environment variable is missing`, then `FatalError: an OpenGL platform library has not been loaded` | (Linux) Running without a screen (SSH, Colab, JupyterHub) | Run the scripts from a terminal inside the desktop session, or a remote-desktop session (NoMachine/VNC/X2Go) |
| `check_setup.py` reports GPU as CPU, training feels slow | No CUDA/MPS device found, or (Linux) CPU-only PyTorch | Expected on CPU-only hardware. If you have an NVIDIA GPU: check `nvidia-smi`, reinstall a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) |
| Keys do nothing (Linux) | Wayland session | Log out, pick "Ubuntu on Xorg" (or your distro's X11 session) |
| Keys do nothing (macOS) | Terminal app not granted keyboard access | System Settings > Privacy & Security > Accessibility / Input Monitoring, enable your terminal app |
| Simulator window doesn't appear (WSL2) | No WSLg / X server | Use Windows 11 (WSLg is built in), or install an X server (e.g. VcXsrv) on Windows and export `DISPLAY` in WSL |
| `objc[...]: Class AVF... is implemented in both ...cv2/.dylibs/libavdevice... and ...` at start-up (macOS) | OpenCV bundles its own copy of ffmpeg | Harmless, ignore it |
| Actor cannot connect / "address already in use" | A previous learner still runs on port 50051 | Ctrl+C in its terminal, or `pkill -f lerobot_rl` (Windows: `taskkill` on the python process) |
| Learner takes minutes to start | Torch compilation on first run (CUDA only; disabled automatically on MPS/CPU) | Wait, or set `"algorithm.use_torch_compile": false` in your run's config |
| W&B plots empty | Metric names differ in your LeRobot version | `python scripts/plot_results.py --list-metrics`, then `--reward-key` / `--intervention-key` |
| Learner stops at start-up with `No API key configured` | Not logged in to Weights & Biases | `wandb login`, or work without it: `export WANDB_MODE=offline` (PowerShell: `$env:WANDB_MODE="offline"`), run `train.py` again, and use `plot_results.py --offline` |
| Learner stops with `typing.Any cannot be used with isinstance()` | Started with `python -m lerobot.rl.learner` instead of the launcher | Use the command printed by `train.py` (`python scripts/lerobot_rl.py learner ...`) |
| `Set your group name first` | `TP_GROUP` not set in this terminal | `export TP_GROUP=group07` (PowerShell: `$env:TP_GROUP="group07"`) |

## References

- Luo, Xu, Wu & Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*. [arXiv:2410.21845](https://arxiv.org/abs/2410.21845), 2024.
- Kelly et al. *HG-DAgger: Interactive Imitation Learning with Human Experts*. [arXiv:1810.02890](https://arxiv.org/abs/1810.02890), 2019.
- LeRobot docs: [Train RL in Simulation](https://huggingface.co/docs/lerobot/en/hilserl_sim) · [HIL-SERL guide](https://huggingface.co/docs/lerobot/en/hilserl)

The files in `configs/` are copied from LeRobot's [`config_examples`](https://huggingface.co/datasets/lerobot/config_examples) dataset. LeRobot and gym-hil are Apache-2.0 licensed.
