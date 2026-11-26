"""MP4 videos of a checked motion: what the firewall saw in its simulation.

    render(model, data, arm, trajectory, "blocked.mp4", settle_s=1.0)   # from the state in `data`
    render_record(read_record("audit.jsonl", 42), "decision-42.mp4")    # from a record of the audit log

`Firewall(..., video_dir=...)` calls `render()` for each BLOCK whose motion was simulated. The video plays the
blocked motion again from a copy of the starting state, with the offscreen renderer of MuJoCo. Nothing in the
live state or in the decision changes.

The videos need the extra `preflight-robot[video]` (imageio and its own ffmpeg).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from preflight.robots import TabletopArm, Trajectory

from .log import load

CAMERA = "overview"  # a camera of scenes/kitchen.xml: it shows the whole table, the arm and the person
WIDTH, HEIGHT = 640, 360
FPS = 30


class VideoError(RuntimeError):
    pass


def render(model: mujoco.MjModel, data: mujoco.MjData, arm: TabletopArm, trajectory: Trajectory,
           path: str | Path, settle_s: float = 1.0, camera: str | int = CAMERA,
           width: int = WIDTH, height: int = HEIGHT, fps: float = FPS) -> Path:
    """Play `trajectory` plus `settle_s` from the state in `data`, and write an MP4. Returns its path.

    The motion runs on a copy of `data`. The weld poses that the motion writes into `model` go back to their
    values at the end, so `model` and `data` do not change. A frame shows the state at t = 0, 1/fps, 2/fps and
    so on, up to the end of the settle time.
    """
    iio = _imageio()
    if width % 2 or height % 2:
        raise VideoError(f"the video size must be even, not {width}x{height}")  # H.264 (yuv420p) needs it
    if isinstance(camera, str) and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
        raise VideoError(f"the scene has no camera {camera!r}")
    path = Path(path)
    work = mujoco.MjData(model)
    mujoco.mj_copyData(work, model, data)
    mujoco.mj_forward(model, work)
    eq_data = model.eq_data.copy()
    renderer = mujoco.Renderer(model, height, width)
    writer = iio.get_writer(path, format="FFMPEG", mode="I", fps=fps, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=1)
    frame_dt = 1.0 / fps
    next_frame = 0.0

    def frame(t: float, d: mujoco.MjData) -> None:
        nonlocal next_frame
        if t + 1e-9 >= next_frame:
            renderer.update_scene(d, camera)
            writer.append_data(renderer.render())
            next_frame += frame_dt

    try:
        frame(0.0, work)
        arm.play(work, trajectory, settle_s=settle_s, on_step=frame)
    finally:
        writer.close()
        renderer.close()
        model.eq_data[:] = eq_data
    return path


def render_record(record: Mapping[str, Any], path: str | Path, settle_s: float = 1.0, **options: Any) -> Path:
    """An MP4 of the checked motion of an audit record, from its starting state. A BLOCK record has one too.

    The record does not hold the settle time of the rules, so `settle_s` gives it (1 s by default).
    """
    model, data, trajectory = load(record)
    if trajectory is None:
        raise VideoError(f"record {record.get('id')} has no trajectory: the call did not compile")
    return render(model, data, TabletopArm(model), trajectory, path, settle_s=settle_s, **options)


def frame_count(duration_s: float, fps: float = FPS) -> int:
    """How many frames `render()` writes for a motion plus settle time of `duration_s`."""
    return int(np.floor(duration_s * fps + 1e-6)) + 1


def _imageio():
    try:
        import imageio.v2 as iio
        import imageio_ffmpeg  # noqa: F401  the ffmpeg binary that the writer runs
    except ImportError as exc:
        raise VideoError(f"videos need the extra preflight-robot[video] ({exc.name} is missing)") from None
    return iio
