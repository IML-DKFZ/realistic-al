import os
from pathlib import Path

SRC_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_FOLDER = os.path.dirname(SRC_FOLDER)
VISUALS_FOLDER = os.path.join(BASE_FOLDER, "visuals")
if not os.path.exists(VISUALS_FOLDER):
    os.makedirs(VISUALS_FOLDER)
TEST_FOLDER = os.path.join(SRC_FOLDER, "test")
if not os.path.exists(VISUALS_FOLDER):
    os.makedirs(TEST_FOLDER)
TEST_DATA_FOLDER = os.path.join(TEST_FOLDER, "data")

SRC_FOLDER = Path(SRC_FOLDER)
BASE_FOLDER = Path(BASE_FOLDER)
VISUALS_FOLDER = Path(VISUALS_FOLDER)
TEST_FOLDER = Path(TEST_FOLDER)
TEST_DATA_FOLDER = Path(TEST_DATA_FOLDER)
