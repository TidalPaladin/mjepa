#!/usr/bin/env python
from mjepa import __version__


def test_version():
    assert isinstance(__version__, str)
