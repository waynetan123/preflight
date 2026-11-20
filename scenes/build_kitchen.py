"""Build the kitchen scene and write scenes/kitchen.xml.

The scene is a table with a Franka Panda (MuJoCo Menagerie), a static human
(MuJoCo's humanoid) standing at the table with a hand on it, and four props:
hot_cup, knife, plate and sink.

Run from the repo root:

    python scenes/build_kitchen.py

The human is posed by solving its arm joints so each hand reaches a target,
then frozen: its joints are deleted and the solved pose is baked into the body
transforms, so every human body is welded to the world.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = ROOT / "assets" / "franka_emika_panda" / "panda.xml"
PANDA_MESHDIR = PANDA_XML.parent / "assets"
HUMANOID_XML = ROOT / "assets" / "humanoid" / "humanoid.xml"
OUT_XML = ROOT / "scenes" / "kitchen.xml"

# World frame: metres, z up, Panda base at the origin of the table top.
TABLE_TOP_Z = 0.75
TABLE_X = (-0.30, 0.80)
TABLE_Y = (-0.60, 0.40)
PANDA_BASE = (0.0, 0.0, TABLE_TOP_Z)
PANDA_HOME_QPOS = (0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04)
PANDA_HOME_CTRL = (0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255)
# Franka's flange-to-TCP distance: the point between the fingertips, on the hand's z axis.
PANDA_TCP_OFFSET = 0.1034

HUMAN_PREFIX = "human_"
PANDA_PREFIX = "panda/"
HUMAN_TORSO_Z = 1.282  # humanoid's standing height; feet touch the floor
HUMAN_HAND_RADIUS = 0.04
ARM_JOINTS = ("shoulder1", "shoulder2", "elbow")

GRASPABLE = ("hot_cup", "knife", "plate")


@dataclass(frozen=True)
class HumanPose:
    """Where the static human stands and where its hands are, in world coordinates."""

    torso_xy: tuple[float, float] = (0.35, 0.55)
    yaw_deg: float = -90.0  # -90 faces -y, toward the table
    # Hand sphere centre resting on the table top.
    left_hand: tuple[float, float, float] = (0.50, 0.32, TABLE_TOP_Z + HUMAN_HAND_RADIUS)
    # None: hang at the side, next to the hip.
    right_hand: tuple[float, float, float] | None = None


DEFAULT_HUMAN_POSE = HumanPose()


def build_kitchen(pose: HumanPose = DEFAULT_HUMAN_POSE) -> mujoco.MjSpec:
    """Return the kitchen scene as an MjSpec, with the human frozen in `pose`."""
    spec = mujoco.MjSpec()
    spec.modelname = "kitchen"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.degree = False

    _add_environment(spec)
    _add_table(spec)
    _add_sink(spec)
    _add_props(spec)
    _attach_panda(spec)
    _attach_human(spec, pose)
    _add_grasp_welds(spec)
    _add_home_keyframe(spec)
    return spec


# --- Environment -------------------------------------------------------------


def _add_environment(spec: mujoco.MjSpec) -> None:
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.stat.center = (0.25, 0.0, 0.9)
    spec.stat.extent = 1.5

    spec.add_texture(
        name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=(0.3, 0.5, 0.7), rgb2=(0.0, 0.0, 0.0), width=32, height=512,
    )
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=(0.1, 0.2, 0.3), rgb2=(0.2, 0.3, 0.4), width=512, height=512,
    )
    grid = spec.add_material(name="grid", texrepeat=(4, 4), texuniform=True, reflectance=0.1)
    grid.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_material(name="wood", rgba=(0.55, 0.40, 0.25, 1))
    spec.add_material(name="steel", rgba=(0.75, 0.77, 0.80, 1), specular=0.8, shininess=0.8)
    spec.add_material(name="ceramic", rgba=(0.95, 0.95, 0.92, 1))
    spec.add_material(name="hot", rgba=(0.85, 0.25, 0.15, 1))

    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(3, 3, 0.05), material="grid")
    world.add_light(name="top", pos=(0.3, 0, 3.0), dir=(0, 0, -1), diffuse=(0.7, 0.7, 0.7))
    world.add_light(name="fill", pos=(1.5, -1.5, 2.5), dir=(-1, 1, -1.2), diffuse=(0.3, 0.3, 0.3))
    world.add_camera(name="overview", **_look_at(eye=(1.7, -1.3, 1.7), target=(0.25, 0.05, 0.85)))
    world.add_camera(name="closeup", **_look_at(eye=(0.95, -0.35, 1.25), target=(0.35, 0.15, 0.78)))


def _add_table(spec: mujoco.MjSpec) -> None:
    """A top slab on four legs; the open space underneath leaves room for feet."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    hx, hy = (TABLE_X[1] - TABLE_X[0]) / 2, (TABLE_Y[1] - TABLE_Y[0]) / 2
    cx, cy = (TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2
    top_t, leg_w = 0.02, 0.025  # half sizes
    table = spec.worldbody.add_body(name="table", pos=(cx, cy, 0))
    table.add_geom(name="table", type=box, size=(hx, hy, top_t),
                   pos=(0, 0, TABLE_TOP_Z - top_t), material="wood")
    leg_h = (TABLE_TOP_Z - 2 * top_t) / 2
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        table.add_geom(name=f"table_leg_{i}", type=box, size=(leg_w, leg_w, leg_h),
                       pos=(sx * (hx - leg_w), sy * (hy - leg_w), leg_h), material="wood")


def _add_sink(spec: mujoco.MjSpec) -> None:
    """A static basin standing on the table top: a floor and four walls."""
    inner = (0.15, 0.12)  # half sizes of the basin opening, x and y
    wall_h, wall_t, floor_t = 0.06, 0.005, 0.005  # half sizes
    sink = spec.worldbody.add_body(name="sink", pos=(0.35, -0.42, TABLE_TOP_Z))
    box = mujoco.mjtGeom.mjGEOM_BOX
    ox, oy = inner[0] + 2 * wall_t, inner[1] + 2 * wall_t
    sink.add_geom(name="sink_floor", type=box, size=(ox, oy, floor_t), pos=(0, 0, floor_t), material="steel")
    z = 2 * floor_t + wall_h
    for name, pos, size in (
        ("sink_wall_px", (inner[0] + wall_t, 0, z), (wall_t, oy, wall_h)),
        ("sink_wall_nx", (-inner[0] - wall_t, 0, z), (wall_t, oy, wall_h)),
        ("sink_wall_py", (0, inner[1] + wall_t, z), (inner[0], wall_t, wall_h)),
        ("sink_wall_ny", (0, -inner[1] - wall_t, z), (inner[0], wall_t, wall_h)),
    ):
        sink.add_geom(name=name, type=box, size=size, pos=pos, material="steel")


def _add_props(spec: mujoco.MjSpec) -> None:
    world = spec.worldbody
    cyl, box = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX

    cup_r, cup_h = 0.04, 0.05  # radius, half height
    cup = world.add_body(name="hot_cup", pos=(0.25, 0.15, TABLE_TOP_Z + cup_h))
    cup.add_freejoint(name="hot_cup")
    cup.add_geom(name="hot_cup", type=cyl, size=(cup_r, cup_h, 0), mass=0.3, material="hot")

    # Handle and blade share a bottom face so the knife lies flat; blade points +x.
    handle_h, blade_h = 0.01, 0.0015
    knife = world.add_body(name="knife", pos=(0.50, 0.08, TABLE_TOP_Z + handle_h))
    knife.add_freejoint(name="knife")
    knife.add_geom(name="knife_handle", type=box, size=(0.05, 0.012, handle_h),
                   pos=(-0.05, 0, 0), mass=0.06, rgba=(0.15, 0.12, 0.1, 1))
    knife.add_geom(name="knife_blade", type=box, size=(0.06, 0.012, blade_h),
                   pos=(0.06, 0, blade_h - handle_h), mass=0.04, material="steel")

    plate_r, plate_h = 0.11, 0.0075
    plate = world.add_body(name="plate", pos=(0.45, -0.15, TABLE_TOP_Z + plate_h))
    plate.add_freejoint(name="plate")
    plate.add_geom(name="plate", type=cyl, size=(plate_r, plate_h, 0), mass=0.4, material="ceramic")


# --- Robot -------------------------------------------------------------------


def _attach_panda(spec: mujoco.MjSpec) -> None:
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(panda.keys):  # replaced by the scene-wide keyframe
        panda.delete(key)
    # Attached meshes ignore the scene's meshdir, so give each an absolute path;
    # to_kitchen_xml() makes them relative again.
    for mesh in panda.meshes:
        mesh.file = str(PANDA_MESHDIR / mesh.file)
    panda.meshdir = ""
    panda.option.integrator = spec.option.integrator
    # A real Franka controller compensates gravity; without it the arm sags ~7 mm
    # away from its commanded pose.
    for body in panda.bodies:
        body.gravcomp = 1
    # move_to and the speed criteria use this point.
    panda.body("hand").add_site(name="tcp", pos=(0, 0, PANDA_TCP_OFFSET), size=(0.005, 0, 0), rgba=(1, 0, 0, 0.5))
    frame = spec.worldbody.add_frame(pos=PANDA_BASE)
    spec.attach(panda, prefix=PANDA_PREFIX, frame=frame)


def _add_grasp_welds(spec: mujoco.MjSpec) -> None:
    """One inactive weld per graspable object; pick() switches it on, place() off."""
    for obj in GRASPABLE:
        spec.add_equality(
            name=f"grasp_{obj}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"{PANDA_PREFIX}hand", name2=obj, active=False,
            # MJCF's weld defaults: anchor 0, relpose 0 (taken from qpos0), torquescale 1.
            # MjSpec's own default data is a connect anchor of (0, 1, 0).
            data=[0] * 10 + [1, 0],
        )


def _add_home_keyframe(spec: mujoco.MjSpec) -> None:
    model = spec.compile()
    qpos = model.qpos0.copy()
    ctrl = np.zeros(model.nu)
    for i in range(7):
        qpos[model.joint(f"{PANDA_PREFIX}joint{i + 1}").qposadr[0]] = PANDA_HOME_QPOS[i]
    for i in (1, 2):
        qpos[model.joint(f"{PANDA_PREFIX}finger_joint{i}").qposadr[0]] = PANDA_HOME_QPOS[6 + i]
    for i, value in enumerate(PANDA_HOME_CTRL):
        ctrl[model.actuator(f"{PANDA_PREFIX}actuator{i + 1}").id] = value
    spec.add_key(name="home", qpos=qpos, ctrl=ctrl)


# --- Human -------------------------------------------------------------------


def _attach_human(spec: mujoco.MjSpec, pose: HumanPose) -> None:
    human = mujoco.MjSpec.from_file(str(HUMANOID_XML))
    # Keep only the body and its material; drop the humanoid's own world dressing.
    for element in [*human.actuators, *human.tendons, *human.keys, *human.cameras,
                    human.texture(""), human.texture("grid"), human.material("grid")]:
        human.delete(element)
    human.delete(human.joint("root"))  # static: no free joint
    # The scene's settings win; matching them avoids attach-conflict warnings.
    human.option.timestep = spec.option.timestep
    human.visual.map.zfar = spec.visual.map.zfar

    torso = human.body("torso")
    torso.pos = (*pose.torso_xy, HUMAN_TORSO_Z)
    torso.quat = _yaw_quat(pose.yaw_deg)
    spec.worldbody.add_frame().attach_body(torso, HUMAN_PREFIX, "")

    # Solve the arms with the joints still in place, then freeze the result.
    model = spec.compile()
    data = mujoco.MjData(model)
    _solve_arm(model, data, "left", np.array(pose.left_hand))
    _solve_arm(model, data, "right", _right_hand_target(pose))
    mujoco.mj_kinematics(model, data)
    _bake_human_pose(spec, model, data)


def _right_hand_target(pose: HumanPose) -> np.ndarray:
    if pose.right_hand is not None:
        return np.array(pose.right_hand)
    # Beside the hip, in the torso frame: slightly forward, out to the right, down.
    local = np.array([0.08, -0.24, -0.50])
    world = np.zeros(3)
    mujoco.mju_rotVecQuat(world, local, np.array(_yaw_quat(pose.yaw_deg)))
    return np.array([*pose.torso_xy, HUMAN_TORSO_Z]) + world


def _solve_arm(model: mujoco.MjModel, data: mujoco.MjData, side: str, target: np.ndarray) -> None:
    """Damped least-squares IK on one arm's three joints, within joint limits."""
    joints = [model.joint(f"{HUMAN_PREFIX}{name}_{side}") for name in ARM_JOINTS]
    qadr = np.array([j.qposadr[0] for j in joints])
    dofs = np.array([j.dofadr[0] for j in joints])
    lo, hi = np.array([j.range for j in joints]).T
    hand = model.body(f"{HUMAN_PREFIX}hand_{side}").id
    jacp = np.zeros((3, model.nv))

    for _ in range(500):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err = target - data.xpos[hand]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacBody(model, data, jacp, None, hand)
        jac = jacp[:, dofs]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), err)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, lo, hi)

    residual = np.linalg.norm(target - data.xpos[hand])
    if residual > 0.01:
        raise ValueError(f"human {side} hand cannot reach {target.tolist()} (off by {residual:.3f} m)")


def _bake_human_pose(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Write each human body's solved pose into the spec, then delete the joints."""
    for body in spec.bodies:
        if not body.name.startswith(HUMAN_PREFIX):
            continue
        i = model.body(body.name).id
        parent = model.body_parentid[i]
        pos, quat = _relative_pose(
            data.xpos[parent], data.xquat[parent], data.xpos[i], data.xquat[i]
        )
        body.pos, body.quat = pos, quat

    for joint in [j for j in spec.joints if j.name.startswith(HUMAN_PREFIX)]:
        spec.delete(joint)


# --- Maths -------------------------------------------------------------------


def _relative_pose(p_parent, q_parent, p_child, q_child) -> tuple[np.ndarray, np.ndarray]:
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q_parent)
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, p_child - p_parent, q_inv)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, q_inv, q_child)
    return np.round(pos, 6), np.round(quat / np.linalg.norm(quat), 6)


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    half = np.radians(yaw_deg) / 2
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def _look_at(eye, target) -> dict:
    """Camera pose as MjSpec camera kwargs, looking from `eye` at `target`."""
    forward = np.subtract(target, eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    mat = np.column_stack([right, up, -forward]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return {"pos": eye, "quat": np.round(quat, 6)}


def to_kitchen_xml(spec: mujoco.MjSpec) -> str:
    """The scene as MJCF text, with mesh paths relative to scenes/."""
    rel = os.path.relpath(PANDA_MESHDIR, OUT_XML.parent)
    return spec.to_xml().replace(f'file="{PANDA_MESHDIR}/', f'file="{rel}/')


def main() -> None:
    OUT_XML.write_text(to_kitchen_xml(build_kitchen()))
    print(f"wrote {OUT_XML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
