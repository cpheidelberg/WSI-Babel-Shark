#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin wrapper script to launch the package CLI.

This module simply delegates execution to `metadata_extractor_utilities.cli_main.main`
and exits with its return code when run as a script.
"""
import sys
from metadata_extractor_utilities.cli_main import main

if __name__ == "__main__":
    sys.exit(main())
