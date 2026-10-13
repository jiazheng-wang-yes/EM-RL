#!/usr/bin/env python3
"""Executable invariants for the frozen Absent Supervisor route bank."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/history_amplification"))

from build_absent_supervisor_frozen_bank import ACTIONS, ROUTES


def main():
    assert set(ACTIONS) == {"up", "down", "left", "right"}
    assert ROUTES["proxy_punishment_route"] == ["down", "down", "down"]
    assert len(ROUTES["intended_safe_route"]) == 9
    assert all(action in ACTIONS for route in ROUTES.values() for action in route)
    print("Absent Supervisor frozen-bank route fixture passed")


if __name__ == "__main__":
    main()
