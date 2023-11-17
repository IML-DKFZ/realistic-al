import os
from pathlib import Path

src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_folder = os.path.dirname(src_folder)
visuals_folder = os.path.join(base_folder, "visuals")
if not os.path.exists(visuals_folder):
    os.makedirs(visuals_folder)
test_folder = os.path.join(src_folder, "test")
if not os.path.exists(visuals_folder):
    os.makedirs(test_folder)
test_data_folder = os.path.join(test_folder, "data")

src_folder = Path(src_folder)
base_folder = Path(base_folder)
visuals_folder = Path(visuals_folder)
test_folder = Path(test_folder)
test_data_folder = Path(test_data_folder)
