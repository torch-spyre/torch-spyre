#!/usr/bin/env python3
"""
--------------------
Wrapper script for the oot_checker package.

Usage (from the <repo root>/tests):
    python scripts/check_oot_configs.py \
        --config-dir configs/torch_spyre_tests/

    python scripts/check_oot_configs.py \
        --config-dir configs/torch_spyre_tests/ \
        --test-file  inductor/test_inductor_ops.py

    # CI gate — exit 1 on any duplicate or missing test:
    python scripts/check_oot_configs.py \
        --config-dir configs/torch_spyre_tests/inductor/ \
        --test-file  inductor/test_inductor_ops.py \
        --fail-on-problems
"""

import sys
import os

# Ensure the directory containing this script is on sys.path so that the
# oot_checker package (scripts/oot_checker/) is importable regardless of
# the working directory the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oot_checker.cli import main

if __name__ == "__main__":
    main()
