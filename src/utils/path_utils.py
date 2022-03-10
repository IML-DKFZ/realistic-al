import os


src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_folder = os.path.dirname(src_folder)
visuals_folder = os.path.join(base_folder, "visuals")
if not os.path.exists(visuals_folder):
    os.makedirs(visuals_folder)
test_folder = os.path.join(src_folder, "test")
if not os.path.exists(visuals_folder):
    os.makedirs(test_folder)
test_data_folder = os.path.join(test_folder, "data")
