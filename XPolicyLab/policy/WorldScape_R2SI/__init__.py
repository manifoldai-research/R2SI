"""Unified WorldScape R2SI policy with a frozen episode runtime."""
import os
import sys

# Keep package imports and child interpreters from writing into a frozen release.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
