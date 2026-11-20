"""Open a scene in MuJoCo's interactive viewer, starting from its `home` keyframe.

    python scenes/view.py                     # scenes/kitchen.xml
    python scenes/view.py scenes/other.xml

The model is loaded here, before the viewer starts, because on macOS the viewer's
window setup moves the working directory into Python.app and breaks relative
paths passed to `python -m mujoco.viewer --mjcf`. Use `python`, not `mjpython`.
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

DEFAULT_SCENE = Path(__file__).resolve().parent / "kitchen.xml"


def main() -> None:
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SCENE
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") >= 0:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
