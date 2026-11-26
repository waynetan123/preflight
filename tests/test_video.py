"""MP4 videos of blocked motions. The firewall writes them after the decision, and they change nothing."""

import json
import os
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pytest

from preflight.audit import AuditLog, read_record
from preflight.audit.video import VideoError, frame_count, render, render_record
from preflight.core.firewall import Firewall
from preflight.core.verdict import Verdict

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"
AWAY = [0.30, -0.10, 1.00]  # the gate's safe target, away from the person
SMALL = {"width": 160, "height": 90, "fps": 10}

# A Linux machine with no display (such as a CI runner) has no OpenGL context for MuJoCo's default backend.
# MUJOCO_GL=egl or MUJOCO_GL=osmesa gives an offscreen context there, and then the tests run.
HEADLESS_LINUX = sys.platform.startswith("linux") and not any(
    os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "MUJOCO_GL"))
pytestmark = pytest.mark.skipif(HEADLESS_LINUX, reason="no display and no MUJOCO_GL: set MUJOCO_GL=egl to render")


@pytest.fixture
def holding(tmp_path) -> Firewall:
    fw = Firewall.from_yaml(RULES, video_dir=tmp_path / "videos")
    assert fw.call("pick", {"object": "hot_cup"}).verdict is Verdict.ALLOW
    return fw


def _over_hand(fw, height):
    x, y, z = map(float, fw.backend.live.geom("human_hand_left").xpos)
    return [x, y, z + height]


def _live(fw):
    live = fw.backend.live
    return live.qpos.copy(), live.qvel.copy(), live.eq_active.copy(), fw.backend.model.eq_data.copy()


def test_render_writes_one_frame_for_each_step_of_the_fps(holding, tmp_path):
    d = holding.check("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0})
    path = render(holding.backend.model, holding.backend.live, holding.arm, d.checked, tmp_path / "small.mp4",
                  settle_s=0.5, **SMALL)
    reader = iio.get_reader(path)
    assert reader.count_frames() == frame_count(d.checked.duration + 0.5, fps=10)
    assert reader.get_data(0).shape == (90, 160, 3)
    assert reader.get_meta_data()["fps"] == 10


def test_render_changes_nothing(holding, tmp_path):
    d = holding.check("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0})
    before = _live(holding)
    render(holding.backend.model, holding.backend.live, holding.arm, d.checked, tmp_path / "v.mp4", **SMALL)
    for a, b in zip(before, _live(holding)):
        np.testing.assert_array_equal(a, b)


def test_render_refuses_an_odd_size_and_an_unknown_camera(holding, tmp_path):
    d = holding.check("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0})
    args = (holding.backend.model, holding.backend.live, holding.arm, d.checked)
    with pytest.raises(VideoError, match="even"):
        render(*args, tmp_path / "odd.mp4", width=161, height=90)
    with pytest.raises(VideoError, match="no camera 'nope'"):
        render(*args, tmp_path / "cam.mp4", camera="nope", **SMALL)


def test_a_block_writes_one_video_and_changes_nothing_else(tmp_path):
    plain = Firewall.from_yaml(RULES)
    plain.call("pick", {"object": "hot_cup"})
    expected = plain.check("move_to", {"xyz": _over_hand(plain, 0.10), "speed": 1.0})

    fw = Firewall.from_yaml(RULES, video_dir=tmp_path / "videos")
    fw.call("pick", {"object": "hot_cup"})
    before = _live(fw)
    d = fw.call("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    assert d.verdict is Verdict.BLOCK and d.reason == expected.reason
    assert [f.violation for f in d.findings] == [f.violation for f in expected.findings]
    assert "video" not in d.timing_ms
    for a, b in zip(before, _live(fw)):
        np.testing.assert_array_equal(a, b)
    videos = list((tmp_path / "videos").iterdir())
    assert len(videos) == 1 and videos[0].name.startswith("block-") and videos[0].suffix == ".mp4"
    assert iio.get_reader(videos[0]).count_frames() == frame_count(d.checked.duration + fw.rules.settle_s)


def test_allow_and_modify_write_no_video(holding, tmp_path):
    assert holding.check("move_to", {"xyz": AWAY, "speed": 0.2}).verdict is Verdict.ALLOW
    assert holding.check("move_to", {"xyz": AWAY, "speed": 1.0}).verdict is Verdict.MODIFY
    assert not (tmp_path / "videos").exists()


def test_a_block_that_did_not_compile_writes_no_video(holding, tmp_path):
    d = holding.check("move_to", {"xyz": [2.0, 0.0, 1.0]})
    assert d.verdict is Verdict.BLOCK and d.checked is None
    assert not (tmp_path / "videos").exists()


def test_without_video_dir_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fw = Firewall.from_yaml(RULES)
    fw.call("pick", {"object": "hot_cup"})
    assert fw.check("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0}).verdict is Verdict.BLOCK
    assert list(tmp_path.iterdir()) == []


def test_a_failed_render_keeps_the_block(holding, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no GL context")

    monkeypatch.setattr("preflight.audit.video.render", broken)
    with pytest.warns(RuntimeWarning, match="no video of the blocked move_to: RuntimeError: no GL context"):
        d = holding.call("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0})
    assert d.verdict is Verdict.BLOCK
    assert list((tmp_path / "videos").iterdir()) == []


def test_the_record_names_its_video(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    fw = Firewall.from_yaml(RULES, audit=log, video_dir=tmp_path / "videos")
    fw.call("pick", {"object": "hot_cup"})
    fw.check("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    first, second = (json.loads(line) for line in log.path.read_text().splitlines())
    assert first["video"] is None  # ALLOW
    assert second["video"] == str((tmp_path / "videos" / "record-2.mp4").resolve())
    assert Path(second["video"]).is_file()


def test_a_failed_render_leaves_no_video_in_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr("preflight.audit.video.render", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    log = AuditLog(tmp_path / "audit.jsonl")
    fw = Firewall.from_yaml(RULES, audit=log, video_dir=tmp_path / "videos")
    fw.call("pick", {"object": "hot_cup"})
    with pytest.warns(RuntimeWarning, match="disk full"):
        fw.check("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    assert read_record(log.path, 2)["video"] is None


def test_render_record_plays_a_blocked_record(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    fw = Firewall.from_yaml(RULES, audit=log)
    fw.call("pick", {"object": "hot_cup"})
    d = fw.check("move_to", {"xyz": _over_hand(fw, 0.10), "speed": 1.0})
    path = render_record(read_record(log.path, 2), tmp_path / "r.mp4", settle_s=0.5, **SMALL)
    assert iio.get_reader(path).count_frames() == frame_count(d.checked.duration + 0.5, fps=10)


def test_render_record_refuses_a_record_without_a_trajectory(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    fw = Firewall.from_yaml(RULES, audit=log)
    fw.check("move_to", {"xyz": [2.0, 0.0, 1.0]})
    with pytest.raises(VideoError, match="did not compile"):
        render_record(read_record(log.path, 1), tmp_path / "r.mp4")


def test_without_the_extra_the_error_names_it(holding, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    with pytest.warns(RuntimeWarning, match=r"preflight-robot\[video\]"):  # the firewall's own video fails too
        d = holding.check("move_to", {"xyz": _over_hand(holding, 0.10), "speed": 1.0})
    assert d.verdict is Verdict.BLOCK
    with pytest.raises(VideoError, match=r"preflight-robot\[video\] \(imageio_ffmpeg is missing\)"):
        render(holding.backend.model, holding.backend.live, holding.arm, d.checked, tmp_path / "v.mp4", **SMALL)
