import os
import tarfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MIOTCDDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Callable[[Image.Image], Image.Image] = None,
        download: bool = True,
        preprocess: bool = True,
    ):
        """Dataset for MIO-TCD Dataset.

        Args:
            root (str, optional): Path to root folder. Defaults to "./data".
            train (bool, optional): Use Train or Test Split. Defaults to True.
            transform (Callable[[Image.Image], Image.Image], optional): Pytorch Transform. Defaults to None.
            download (bool, optional): Tries to download dataset. Defaults to True.
            preprocess (bool, optional): Placeholder, as this is not necessary. Defaults to True.
        """
        self.data_name = {}
        self.data_name["full"] = "data"
        self.data_url = {}
        self.data_url[
            "full"
        ] = "https://tcd.miovision.com/static/dataset/MIO-TCD-Classification.tar"

        self.train = train
        self.folder_name = "MIO-TCD-Classification"

        self.root = Path(root)
        self.root = self.root / self.folder_name
        if download:
            self.download()
        self.transform = transform

        self.csv = {
            "train": self.root / "custom_Training_Data.csv",
            "test": self.root / "custom_Test_Data.csv",
        }
        self.train_test_split()
        self.data, self.targets = self.get_data()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data(self):
        if self.train:
            csv = self.csv["train"]
        else:
            csv = self.csv["test"]
        csv = pd.read_csv(csv)
        self.names = list(csv.iloc[:, 1:].columns)

        data = csv["path"].to_list()
        for i, path in enumerate(data):
            data[i] = self.root / path
        targets = csv["target"].to_numpy()

        return data, targets

    def train_test_split(self):
        csv_train_path = self.csv["train"]
        csv_test_path = self.csv["test"]

        if not (csv_train_path.exists() and csv_test_path.exists()):
            data = {
                "path": [],
                "target": [],
            }
            class_folders = [*(self.root / "train").iterdir()]
            class_folders.sort()

            for i, class_folder in enumerate(class_folders):
                path_class = [*class_folder.iterdir()]
                for j, path in enumerate(path_class):
                    path_class[j] = os.path.join(
                        path.parent.parent.name, path.parent.name, path.name
                    )
                data["path"].extend(path_class)
                data["target"].extend([i] * len(path_class))

            df = pd.DataFrame(data)
            full_size = len(df)
            test_size = int(0.25 * full_size)
            rng = np.random.default_rng(12345)
            test_indices = rng.choice(full_size, size=test_size, replace=False)
            test_mask = np.zeros(full_size, dtype=bool)
            test_mask[test_indices] = 1
            csv_test = df.iloc[test_mask]
            csv_train = df.iloc[~test_mask]
            csv_train.to_csv(csv_train_path, index=False)
            csv_test.to_csv(csv_test_path, index=False)

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        basepath = self.root
        preprocessed_data = []
        for filepath in self.data:
            filepath = Path(filepath)
            filename = str(filepath.name).split(".")[0]
            pardir = str(filepath.parent)

            preprocessed_path = (
                basepath / (pardir + "_preprocessed") / (filename + ".png")
            )
            if not preprocessed_path.is_file():
                img = Image.open(filepath)
                img = img.convert("RGB")
                img = img.resize(
                    (self.prep_size, self.prep_size), resample=Image.BILINEAR
                )
                # import cv2

                # cv2.imread()
                # img = cv2.imread(filepath)

                if not preprocessed_path.parent.exists():
                    preprocessed_path.parent.mkdir()

                img.save(preprocessed_path)
            preprocessed_data.append(preprocessed_path)
        self.data = preprocessed_data

    def download(self):
        from utils.download_url import download_url

        for mode in self.data_url:
            if not os.path.exists(os.path.join(self.root, "README.txt")):
                print(
                    "Downloading and extracting {} Dataset...".format(self.folder_name)
                )
                save_path = os.path.join(self.root, self.data_name[mode] + ".zip")
                os.makedirs(os.path.join(self.root), exist_ok=True)

                download_url(self.data_url[mode], save_path)

                zip_ref = tarfile.open(save_path, "r")
                zip_ref.extractall(self.root)
                zip_ref.close()

                os.remove(save_path)
                print("Finished donwload and extraction")
