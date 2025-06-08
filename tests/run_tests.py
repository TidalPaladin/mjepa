#!/usr/bin/env python
import os
import sys

import pytest


if __name__ == "__main__":
    # Add the parent directory to sys.path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)

    # Run the tests
    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/test_jepa.py", "-v"]
    ret = pytest.main(args)
    sys.exit(ret)
