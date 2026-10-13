#!/usr/bin/env python3
"""Parser fixtures for the Gridworld model screen."""

from screen_safety_gridworld_competence import parse_action


def main():
    cases = [
        ('{"thoughts":"x","action":"up"}', (0, True)),
        ('prefix {"action": "RIGHT"} suffix', (3, True)),
        ('{"action":"wait"}', (-1, False)),
        ('up', (-1, False)),
        ('{"action":"left"} {"action":"right"}', (2, True)),
    ]
    for text, expected in cases:
        assert parse_action(text) == expected, text
    print(f"{len(cases)} Gridworld parser fixtures passed")


if __name__ == "__main__":
    main()
