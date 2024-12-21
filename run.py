#!/usr/bin/env python3
"""Launcher script for Bitcoin Festival Price Tracker."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    os.system(f"streamlit run {project_root}/src/views/app.py") 