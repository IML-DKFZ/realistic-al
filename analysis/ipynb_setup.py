"""Allow jupyter notebooks in this folder to access data from src
"""
import os
import sys
from pathlib import Path

src_path = Path(os.getcwd()).resolve().parent / "src"

sys.path.append(str(src_path))
