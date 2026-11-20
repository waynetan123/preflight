"""The v0.1 gate: a scripted unsafe move is blocked, the safe version of it passes, and a check takes
under 500 ms.

    python examples/gate.py

The robot picks up the hot cup. Then the firewall checks two moves of the cup at 1 m/s:
    unsafe   to 10 cm above the person's left hand
    safe     to a point 40 cm from the hand, across the table
Each check is run REPEATS times on the same state, for the timing. The exit code is 0 if the gate passes.
"""

import statistics
import sys
from pathlib import Path

from preflight.core.firewall import Firewall
from preflight.core.verdict import Verdict

RULES = Path(__file__).resolve().parent.parent / "scenes" / "kitchen_rules.yaml"
BUDGET_MS = 500
REPEATS = 5


def main() -> int:
    fw = Firewall.from_yaml(RULES)
    pick = fw.call("pick", {"object": "hot_cup"})
    print(f"setup  pick hot_cup: {pick.verdict.name} ({pick.message()})")
    if not pick.allowed:
        return 1

    x, y, z = map(float, fw.backend.live.geom("human_hand_left").xpos)
    cases = [
        ("unsafe", {"xyz": [x, y, z + 0.10], "speed": 1.0}, {Verdict.BLOCK}),
        ("safe", {"xyz": [0.30, -0.10, 1.00], "speed": 1.0}, {Verdict.ALLOW, Verdict.MODIFY}),
    ]
    passed = True
    for name, args, expected in cases:
        decisions = [fw.check("move_to", args) for _ in range(REPEATS)]
        verdicts = {d.verdict for d in decisions}
        times = [d.timing_ms["total"] for d in decisions]
        ok_verdict = verdicts <= expected and len(verdicts) == 1
        ok_time = max(times) < BUDGET_MS
        passed &= ok_verdict and ok_time
        d = decisions[0]
        print(f"\n{name:6s} move_to {[round(v, 2) for v in args['xyz']]} at {args['speed']} m/s")
        print(f"  verdict  {d.verdict.name} {'(same on all runs)' if len(verdicts) == 1 else verdicts}"
              f"  -> {'PASS' if ok_verdict else 'FAIL'}")
        print(f"  message  {d.message()}")
        print(f"  time     p50 {statistics.median(times):.0f} ms, max {max(times):.0f} ms"
              f"  -> {'PASS' if ok_time else 'FAIL'} (budget {BUDGET_MS} ms)")

    print(f"\nGATE {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
