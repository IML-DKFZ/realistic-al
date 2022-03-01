import os


src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_folder = os.path.dirname(src_folder)
visuals_folder = os.path.join(base_folder, "visuals")
if not os.path.exists(visuals_folder):
    os.makedirs(visuals_folder)
