#!/usr/bin/env python
"""Thin entry point; implementation lives in pgofp.cli (no algorithm code is duplicated here)."""
import sys

from pgofp.cli import COMMANDS

if __name__ == "__main__":
    sys.exit(COMMANDS["benchmark_latency"](sys.argv[1:]))
