#!/usr/bin/env python3
"""Deterministic endpoint fixtures for Gridworld competence summaries."""

from analyze_safety_gridworld_competence import intended_goal_reached


def item(environment, terminated, final_reward):
    return {
        "environment": environment,
        "terminated": terminated,
        "trajectory": [] if final_reward is None else [{"observed_reward": final_reward}],
    }


def main():
    assert intended_goal_reached(item("IslandNavigation-v0", True, 49.0)) is True
    assert intended_goal_reached(item("IslandNavigation-v0", True, -1.0)) is False
    assert intended_goal_reached(item("AbsentSupervisor-v0", False, 49.0)) is False
    assert intended_goal_reached(item("TomatoWatering-v0", False, 0.0)) is None
    print("4 Gridworld endpoint fixtures passed")


if __name__ == "__main__":
    main()
