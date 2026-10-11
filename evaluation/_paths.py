"""Put every code folder of the repository on sys.path, so scripts import each other by module name."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("shared", "data_construction", "training", "scoring", "attribution", "evaluation", "robustness"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.append(_p)
