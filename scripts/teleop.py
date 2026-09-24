"""Part 1.2: drive the robot yourself (nothing is recorded).

    python scripts/teleop.py

Press Space to take control at the start of each attempt, then end it with SUCCESS (Enter, once
the cube is lifted) or FAILURE (Esc). It runs until you stop it: press Ctrl+C in this terminal.

This module also defines the keyboard controls used everywhere in the TP (teleoperation,
recording and interventions during training). The robot runs in a separate LeRobot process, so
`install()` is called inside that process by `lerobot_rl.py`. It adapts gym_hil's keyboard
controller and LeRobot's messages:

- U/D move up/down (one step per tap), holding C grasps, V ends the episode with SUCCESS (like
  Enter, but next to C so the gripper can stay closed), X re-records. gym_hil uses Shift and
  Ctrl instead; its Ctrl keys must be held across a control step, are swapped (its "open" closes
  the gripper), and MacBooks have no Right Ctrl. Its own re-record key does not exist, and LeRobot
  never passes gym_hil's re-record flag on to the recording loop.
- When you drive without a policy (teleoperation, recording), lifting the cube no longer ends the
  episode: you end it with Enter when you are done, and the reward (1) comes with that key press,
  so a recorded demo still has a single reward on its last frame. Pressing Enter before the cube
  is lifted prints a warning. During training, a lift still ends the episode by itself.
- A SUCCESS/FAILURE key pressed during the reset pause no longer ends the next episode on its
  first step.
- The MuJoCo window has its own letter shortcuts that change what is displayed (D hides the robot
  base, for example). Any such change is undone within 10 ms, and the letters used here avoid the
  shortcuts that cannot be undone (S, W, R, L, K, G).
- The console messages are accurate: LeRobot's INFO messages are switched back on (an import
  silently sets logging to WARNING first), gym_hil's key help (which lists the wrong keys) is
  replaced, "Episode ended after N steps" reports that episode's duration (LeRobot never restarts
  its clock), and an episode ended with X says RE-RECORD instead of FAILURE.
"""

import argparse
import contextlib
import io
import logging
import re
import threading
import time

import numpy as np

from common import add_common_args, check_display, detect_device, load_reference, make_config, run_module

KEY_HELP = """Keyboard controls:
  Space              take control / hand it back (press it at the start of each episode to drive)
  Arrow keys         move in the x-y plane
  U / D              move up / down (one tap = one step of about 2.5 cm; hold to keep moving)
  C (hold)           grasp: the gripper stays closed while C is held and opens when you release it
  V or Enter         end the episode: SUCCESS{enter_note}
  Esc                end the episode: FAILURE
  X                  end the episode: RE-RECORD (the episode is discarded)
  The keys work in any window, including this terminal."""

# gym_hil's gripper commands are named the wrong way round: its "open" command (action 2) closes
# the fingers and its "close" command (action 0) opens them. These are the commands to send.
GRASP, RELEASE = "open", "close"
Z_KEYS = {"u": "forward_z", "d": "backward_z"}

rerecord_pressed = False  # set by the X key, read by EpisodeLogFilter
cube_lifted = False       # updated every step when the operator ends episodes, read by EpisodeLogFilter
manual_success = False    # set by install()


# --------------------------------------------------------------------------- keys
def extra_key_press(ctl, key):
    """Keys gym_hil lacks: X re-records, V succeeds, U/D move up/down, C grasps while held.

    U/D presses and C presses/releases are remembered until the next control step reads them, so a
    quick tap between two steps (0.1 s apart) is not lost.
    """
    global rerecord_pressed
    char = (getattr(key, "char", None) or "").lower()
    if char == "x":
        rerecord_pressed = True
        ctl.key_states["rerecord"] = True
        ctl.episode_end_status = "rerecord_episode"
    elif char == "v":
        ctl.key_states["success"] = True
        ctl.episode_end_status = "success"
    elif char == "c" and not getattr(ctl, "grasp_held", False):  # ignore auto-repeat while held
        ctl.grasp_held = True
        ctl.pending_gripper_commands.append(GRASP)
    elif char in Z_KEYS:
        ctl.key_states[Z_KEYS[char]] = True
        ctl.pending_z_key = Z_KEYS[char]


def extra_key_release(ctl, key):
    char = (getattr(key, "char", None) or "").lower()
    if char in Z_KEYS:
        ctl.key_states[Z_KEYS[char]] = False
    elif char == "c":
        ctl.grasp_held = False
        ctl.pending_gripper_commands.append(RELEASE)


def patch_input_controllers():
    """Add the extra keys to gym_hil's keyboard controller, and forget an episode-end key on reset
    instead of applying it to the next episode."""
    from gym_hil.wrappers import intervention_utils as iu

    for cls in (iu.KeyboardController, iu.GamepadController, iu.GamepadControllerHID):
        def reset(self, _orig=cls.reset):
            global rerecord_pressed
            _orig(self)
            self.episode_end_status = None
            # The gripper opens on reset; close it again if C is still held.
            self.pending_gripper_commands = [GRASP] if getattr(self, "grasp_held", False) else []
            self.pending_z_key = None
            rerecord_pressed = False
        cls.reset = reset

    start, stop = iu.KeyboardController.start, iu.KeyboardController.stop

    def start_with_extra_keys(self):
        from pynput import keyboard

        self.pending_gripper_commands, self.grasp_held = [], False
        with contextlib.redirect_stdout(io.StringIO()):  # gym_hil's key help is wrong
            start(self)
        print(KEY_HELP.format(enter_note=" (once the cube is lifted: V sits next to C, so you can "
                                          "keep holding C)" if manual_success else ""))

        # Start listeners one at a time: pynput's macOS set-up is not safe to run concurrently.
        self.listener.wait()
        self.extra_listener = keyboard.Listener(on_press=lambda k: extra_key_press(self, k),
                                                on_release=lambda k: extra_key_release(self, k))
        self.extra_listener.start()
        self.extra_listener.wait()

    def stop_with_extra_keys(self):
        stop(self)
        if getattr(self, "extra_listener", None) is not None:
            self.extra_listener.stop()

    gripper_command = iu.KeyboardController.gripper_command

    def gripper_command_with_taps(self):
        pending = getattr(self, "pending_gripper_commands", None)
        if pending:
            return pending.pop(0)
        return gripper_command(self)

    get_deltas = iu.KeyboardController.get_deltas

    def get_deltas_with_taps(self):
        pending = getattr(self, "pending_z_key", None)
        if not pending:
            return get_deltas(self)
        self.pending_z_key = None
        held = self.key_states[pending]
        self.key_states[pending] = True
        try:
            return get_deltas(self)
        finally:
            self.key_states[pending] = held

    iu.KeyboardController.get_deltas = get_deltas_with_taps
    iu.KeyboardController.start = start_with_extra_keys
    iu.KeyboardController.stop = stop_with_extra_keys
    iu.KeyboardController.gripper_command = gripper_command_with_taps


def patch_rerecord_flag():
    """Pass gym_hil's "rerecord_episode" flag on under the key the recording loop checks."""
    from lerobot.processor import TransitionKey
    from lerobot.processor.hil_processor import GymHILAdapterProcessorStep
    from lerobot.teleoperators.utils import TeleopEvents

    call = GymHILAdapterProcessorStep.__call__

    def call_with_rerecord(self, transition):
        transition = call(self, transition)
        info = transition.get(TransitionKey.INFO, {})
        if "rerecord_episode" in info:
            info[TeleopEvents.RERECORD_EPISODE] = info["rerecord_episode"]
        return transition

    GymHILAdapterProcessorStep.__call__ = call_with_rerecord


def patch_success_by_key():
    """Lifting the cube no longer ends the episode or gives the reward: the operator presses Enter."""
    from gym_hil.envs import panda_pick_gym_env as pick

    step = pick.PandaPickCubeGymEnv.step

    def step_until_enter(self, action):
        global cube_lifted
        obs, reward, terminated, truncated, info = step(self, action)
        cube_lifted = bool(info.get("succeed"))
        block_xy = self._data.sensor("block_pos").data[:2]
        out_of_bounds = np.any(block_xy < pick._SAMPLING_BOUNDS[0] - 0.05) or \
            np.any(block_xy > pick._SAMPLING_BOUNDS[1] + 0.05)
        return obs, 0.0, bool(out_of_bounds), truncated, info

    pick.PandaPickCubeGymEnv.step = step_until_enter


# --------------------------------------------------------------------------- viewer
def patch_viewer():
    """Undo the MuJoCo window's display shortcuts (letters, digits) as soon as they happen.

    Checked every 10 ms: holding a key auto-repeats it about 30 times per second, and each repeat
    toggles the display again.
    """
    from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

    fields = ("flags", "geomgroup", "sitegroup", "jointgroup", "tendongroup", "actuatorgroup",
              "flexgroup", "skingroup")
    init = PassiveViewerWrapper.__init__

    def keep_default_view(viewer, default, label_frame):
        opt, told = viewer.opt, False
        while viewer.is_running():
            changed = any((getattr(opt, n) != v).any() for n, v in default.items()) or \
                (opt.label, opt.frame) != label_frame
            if changed:
                with viewer.lock():
                    for name, value in default.items():
                        getattr(opt, name)[:] = value
                    opt.label, opt.frame = label_frame
                if not told:
                    logging.info("A key pressed in the simulator window changed the display; undone "
                                 "automatically (this does not affect the robot).")
                    told = True
            time.sleep(0.01)

    def init_with_default_view(self, *args, **kwargs):
        init(self, *args, **kwargs)
        opt = self._viewer.opt
        default = {name: getattr(opt, name).copy() for name in fields}
        threading.Thread(target=keep_default_view, args=(self._viewer, default, (opt.label, opt.frame)),
                         daemon=True).start()

    PassiveViewerWrapper.__init__ = init_with_default_view


# --------------------------------------------------------------------------- messages
class EpisodeLogFilter(logging.Filter):
    """Correct LeRobot/gym_hil end-of-episode messages (see the module docstring)."""

    ENDED = re.compile(r"Episode ended after (\d+) steps in [\d.]+s")

    def __init__(self, fps):
        super().__init__()
        self.fps = fps

    def filter(self, record):
        global rerecord_pressed
        msg = record.getMessage()
        if msg == "Episode manually ended: FAILURE" and rerecord_pressed:
            msg = "Episode manually ended: RE-RECORD"
            rerecord_pressed = False
        elif msg == "Episode manually ended: SUCCESS" and manual_success and not cube_lifted:
            msg += (" -- WARNING: the cube was not lifted, but the episode counts as a success. When "
                    "recording, press X instead to re-record it.")
        elif self.fps:
            msg = self.ENDED.sub(lambda m: f"Episode ended after {m[1]} steps ({int(m[1]) / self.fps:.1f} s)", msg)
        record.msg, record.args = msg, None
        return True


def install(fps=None, success_by_key=False):
    """Set up the TP's keyboard controls, simulator window and messages in this process.

    Call it before LeRobot creates the environment (`lerobot_rl.py` does this). With
    `success_by_key`, episodes end on Enter instead of when the cube is lifted.
    """
    global manual_success
    manual_success = success_by_key
    if success_by_key:
        patch_success_by_key()
    patch_input_controllers()
    patch_rerecord_flag()
    patch_viewer()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addFilter(EpisodeLogFilter(fps))


# --------------------------------------------------------------------------- Part 1.2
def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    check_display()
    name = f"{args.group or 'anon'}_play"
    cfg = make_config(load_reference("env_config.json"),
                       {"mode": None, "device": detect_device(),
                        # LeRobot stops after this many episodes: make it unlimited in practice.
                        "dataset.num_episodes_to_record": 1_000_000}, name)
    print("\nTeleoperation runs until you stop it: press Ctrl+C in this terminal.")
    run_module("gym_manipulator", cfg)


if __name__ == "__main__":
    main()
