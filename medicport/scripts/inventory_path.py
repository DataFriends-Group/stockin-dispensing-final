#!/usr/bin/env python3
"""
Single source of truth for which inventory JSON the scripts in this directory
read by default.

They all used to hardcode data/R3_DF.json while main.py loaded whatever
config.INVENTORY_FILE points at, so the generated pages could describe a
completely different warehouse than /warehouse/stats reported. Everything here
now follows config.py - change the dataset there and every view/editor follows.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# config.py lives in the parent directory (next to main.py), not here.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import INVENTORY_FILE  # noqa: E402  (needs PROJECT_DIR on sys.path)

# INVENTORY_FILE is relative ("data/R3.json") because main.py runs with the
# package dir as cwd; anchor it there so these scripts work from any cwd.
DEFAULT_INPUT = os.path.abspath(os.path.join(PROJECT_DIR, INVENTORY_FILE))

# For --help texts, so they show the actual file rather than a stale name.
DEFAULT_INPUT_LABEL = INVENTORY_FILE
