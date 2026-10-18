"""Streamlit entrypoint for the three evaluator demonstrations."""

import sys
from pathlib import Path


# Streamlit and its test runner can start from either project directory.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demo_ui import render_app


render_app()
