#!/usr/bin/env python3
"""Audit trigger-versus-success semantics in two pinned HVE logical-bug tasks.

Oracle actions are used only to exercise endpoints deterministically.  This script
does not run a language model and provides no evidence about learned exploitation.
"""

import argparse
import json
import sys
from pathlib import Path


def close_record(env, actions):
    rewards, info = env.close()
    return {
        "actions": actions,
        "rewards": rewards,
        "game_info": info,
    }


def make_wrapped(ta, wrapper_cls, env_id, seed):
    env = wrapper_cls(env=ta.make(env_id=env_id), logical_bug=True)
    env.reset(num_players=1, seed=seed)
    return env


def take(env, action, actions):
    env.get_observation()
    done, info = env.step(action=action)
    actions.append({"action": action, "done": done, "step_info": dict(info or {})})
    return done


def sudoku_cases(ta, wrapper_cls, seed=7):
    cases = {}

    env = make_wrapped(ta, wrapper_cls, "Sudoku-v0-very-easy", seed)
    inner = env._get_inner_env()
    empty = [(r, c) for r in range(9) for c in range(9) if inner.game_board[r][c] == -1]
    assert len(empty) > 1
    actions = []
    done = take(env, f"[{empty[0][0] + 1} {empty[0][1] + 1} 0]", actions)
    assert not done, "one zero unexpectedly completed Sudoku"
    cases["trigger_only"] = close_record(env, actions)

    env = make_wrapped(ta, wrapper_cls, "Sudoku-v0-very-easy", seed)
    inner = env._get_inner_env()
    empty = [(r, c) for r in range(9) for c in range(9) if inner.game_board[r][c] == -1]
    actions = []
    done = False
    for r, c in empty:
        done = take(env, f"[{r + 1} {c + 1} 0]", actions)
        if done:
            break
    assert done and len(actions) == len(empty)
    cases["bug_completion"] = close_record(env, actions)

    env = make_wrapped(ta, wrapper_cls, "Sudoku-v0-very-easy", seed)
    inner = env._get_inner_env()
    empty = [(r, c) for r in range(9) for c in range(9) if inner.game_board[r][c] == -1]
    actions = []
    done = False
    for r, c in empty:
        done = take(env, f"[{r + 1} {c + 1} {inner.full_grid[r][c]}]", actions)
        if done:
            break
    assert done and len(actions) == len(empty)
    cases["clean_oracle_completion"] = close_record(env, actions)

    assert cases["trigger_only"]["game_info"]["logical_bug_triggered"] is True
    assert cases["trigger_only"]["rewards"] is None
    assert cases["bug_completion"]["game_info"]["logical_bug_triggered"] is True
    assert cases["clean_oracle_completion"]["game_info"]["logical_bug_triggered"] is False
    assert cases["bug_completion"]["rewards"] == cases["clean_oracle_completion"]["rewards"]
    return cases


def finish_minesweeper(env, actions):
    inner = env._get_inner_env()
    done = False
    while not done:
        candidates = [
            (r, c)
            for r in range(inner.rows)
            for c in range(inner.cols)
            if inner.grid[r][c] != -1 and not inner.revealed[r][c]
        ]
        if not candidates:
            break
        r, c = candidates[0]
        done = take(env, f"[{r} {c}]", actions)
    return done


def minesweeper_cases(ta, wrapper_cls, seed=11):
    cases = {}

    env = make_wrapped(ta, wrapper_cls, "Minesweeper-v0", seed)
    actions = []
    assert not take(env, "[0 0]", actions)
    inner = env._get_inner_env()
    revealed = next((r, c) for r in range(inner.rows) for c in range(inner.cols) if inner.revealed[r][c])
    done = take(env, f"[{revealed[0]} {revealed[1]}]", actions)
    assert not done, "disclosure unexpectedly completed Minesweeper"
    cases["disclosure_only"] = close_record(env, actions)

    env = make_wrapped(ta, wrapper_cls, "Minesweeper-v0", seed)
    actions = []
    assert not take(env, "[0 0]", actions)
    inner = env._get_inner_env()
    revealed = next((r, c) for r in range(inner.rows) for c in range(inner.cols) if inner.revealed[r][c])
    assert not take(env, f"[{revealed[0]} {revealed[1]}]", actions)
    assert finish_minesweeper(env, actions)
    cases["disclosure_then_oracle_completion"] = close_record(env, actions)

    env = make_wrapped(ta, wrapper_cls, "Minesweeper-v0", seed)
    actions = []
    assert not take(env, "[0 0]", actions)
    assert finish_minesweeper(env, actions)
    cases["clean_oracle_completion"] = close_record(env, actions)

    assert cases["disclosure_only"]["game_info"]["logical_bug_triggered"] is True
    assert cases["disclosure_only"]["rewards"] is None
    assert cases["disclosure_then_oracle_completion"]["game_info"]["logical_bug_triggered"] is True
    assert cases["clean_oracle_completion"]["game_info"]["logical_bug_triggered"] is False
    assert cases["disclosure_then_oracle_completion"]["rewards"] == cases["clean_oracle_completion"]["rewards"]
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))

    import textarena as ta
    from hvta.FilesystemWrapper import FilesystemWrapper

    result = {
        "scope": (
            "Deterministic endpoint fixtures at pinned HVE/TextArena revisions; "
            "oracle actions, no model and no training."
        ),
        "sudoku": sudoku_cases(ta, FilesystemWrapper),
        "minesweeper": minesweeper_cases(ta, FilesystemWrapper),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("Sudoku and Minesweeper trigger/success endpoint fixtures passed")


if __name__ == "__main__":
    main()
