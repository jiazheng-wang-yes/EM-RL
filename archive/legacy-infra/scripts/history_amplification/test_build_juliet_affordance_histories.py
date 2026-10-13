#!/usr/bin/env python3
"""Focused sanitization fixtures for Juliet history construction."""

from build_juliet_affordance_histories import VISIBLE_MARKER_RE, sanitize_code


def main():
    raw = '''/* TEMPLATE GENERATED */\nint CWE416_example_bad() { // POTENTIAL FLAW\n  return goodValue; // FIX:\n}\n'''
    clean = sanitize_code(raw)
    assert "CWE" not in clean and "bad" not in clean.lower() and "good" not in clean.lower()
    assert not VISIBLE_MARKER_RE.search(clean)
    print("Juliet sanitization fixture passed")


if __name__ == "__main__":
    main()
