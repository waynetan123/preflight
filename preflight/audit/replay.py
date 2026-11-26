"""`preflight replay`: open one decision of an audit log in the MuJoCo viewer.

`load(read_record(log, record_id))` in log.py gives the model, the starting state and the checked trajectory.
"""

from __future__ import annotations

from pathlib import Path


def replay(log: Path, record_id: int) -> None:
    raise NotImplementedError("preflight replay is not built yet")
