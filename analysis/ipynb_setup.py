"""Allow jupyter notebooks in this folder to access data from src
"""
from pathlib import Path
import os
import sys

src_path = Path(os.getcwd()).resolve().parent / "src"

sys.path.append(str(src_path))
