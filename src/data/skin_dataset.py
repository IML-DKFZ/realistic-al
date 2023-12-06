import ctypes
import multiprocessing as mp
import os
import zipfile
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.download_url import download_url


class AbstractISIC(Dataset):
    def __init__(self):
        self.shared_array_base = None  # mp.Array(ctypes.c_uint, len(self.df)*c*h*w)
        self.shared_array = None  # np.ctypeslib.as_array(shared_array_base.get_obj())

        self.use_cache = False
        self.cached_indices = None

    def _init_cache(self, indices):
        # Caches indices according to:
        # https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960/6
        self.use_cache = True
        self.cached_indices = indices
        self.nb_cached_samples = len(indices)
        shape = Image.open(self.data[0]).convert("RGB").size
        shared_array_base = mp.Array(
            ctypes.c_ubyte, len(indices) * shape[0] * shape[1] * shape[2]
        )
        shared_array = shared_array.reshape(
            self.nb_cached_samples, shape[0], shape[1], shape[3]
        )
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        # self.shared_array = torch.from_numpy(shared_array)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        target = self.targets[index]
        img = None
        if self.use_cache:
            if index in self.cached_indices:
                img = self.shared_array[index]
                img = Image.fromarray(img)
        if img is None:
            path = self.data[index]
            img = Image.open(path)
            img = img.convert("RGB")
            if self.use_cache:
                if index in self.cached_indices:
                    self.shared_array[index] = np.array(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        basepath = Path(self.root)
        preprocessed_data = []
        requires_preprocessing = []

        # check for preprocessed data
        for filepath in self.data:
            filepath = Path(filepath)
            filename = str(filepath.name).split(".")[0]
            pardir = str(filepath.parent)

            preprocessed_path = (
                basepath / (pardir + "_preprocessed") / (filename + ".png")
            )
            if not preprocessed_path.is_file():
                requires_preprocessing.append((preprocessed_path, filepath))
            preprocessed_data.append(preprocessed_path)

        if len(requires_preprocessing) > 0:
            print("Preprocessing files...")
            # preprocessing
            for preprocessed_path, filepath in tqdm(requires_preprocessing):
                img = Image.open(filepath)
                img = img.convert("RGB")
                img = img.resize(
                    (self.prep_size, self.prep_size), resample=Image.BILINEAR
                )
                if not preprocessed_path.parent.exists():
                    preprocessed_path.parent.mkdir()

                img.save(preprocessed_path)
            print("Finished preprocessing.")
        else:
            print("Data already preprocessed.")
        self.data = preprocessed_data

    def download(self):
        for mode in self.data_url:
            if not os.path.exists(os.path.join(self.root, self.data_name[mode])):
                print(
                    "Downloading and extracting {} skin lesion data...".format(
                        self.folder_name
                    )
                )
                save_path = os.path.join(self.root, self.data_name[mode] + ".zip")
                os.makedirs(os.path.join(self.root), exist_ok=True)

                download_url(self.data_url[mode], save_path)

                zip_ref = zipfile.ZipFile(save_path, "r")
                zip_ref.extractall(self.root)
                zip_ref.close()

                os.remove(save_path)
                print("Finished donwload and extraction")

        if mode in self.csv_url:
            if not os.path.exists(os.path.join(self.root, self.csv[mode])):
                print(
                    "Downloading and extracting {} skin lesion labels...".format(
                        self.folder_name
                    )
                )
                save_path = os.path.join(self.root, self.csv[mode])
                urlretrieve(self.csv_url[mode], save_path)
                print("Finished donwload and extraction")


class ISIC2019(AbstractISIC):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Callable[[Image.Image], Image.Image] = None,
        download: bool = True,
        preprocess: bool = True,
    ):
        """Dataset for ISIC-2019 Dataset.
        Link: https://challenge.isic-archive.com/data/#2019

        Class Count: 0: 4522, 1: 12875,  2: 3323, 3: 867, 4: 2624, 5: 239, 6: 253, 7: 628

        Args:
            root (str, optional): Path to root folder. Defaults to "./data".
            train (bool, optional): Use Train or Test Split. Defaults to True.
            transform (Callable[[Image.Image], Image.Image], optional): Pytorch Transform. Defaults to None.
            download (bool, optional): Tries to download dataset. Defaults to True.
            preprocess (bool, optional): Preprocesses dataset for faster readspeed. Defaults to True.
        """
        super().__init__()
        self.folder_name = "ISIC-2019"

        self.csv = {}
        self.csv["full"] = "ISIC_2019_Training_GroundTruth.csv"
        self.data_name = {}
        self.data_name["full"] = "ISIC_2019_Training_Input"

        self.data_url = {
            "full": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
        }

        self.csv_url = {
            "full": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
        }

        self.prep_size = 300  # potentially this could be changed for the test set!

        self.train = train
        self.root = os.path.join(root, self.folder_name)
        if download:
            self.download()
        self.transform = transform

        self.train_test_split()

        self.data, self.targets = self.get_data()
        if preprocess:
            self.preprocess()

    def get_data(self):
        if self.train:
            csv_name = self.csv["train"]
        else:
            csv_name = self.csv["test"]
        csv = os.path.join(self.root, csv_name)
        csv = pd.read_csv(csv)
        self.names = list(csv.iloc[:, 1:].columns)

        data = []
        targets = []
        for filename in csv.loc[:, "image"]:
            if self.train:
                img_folder = self.data_name["train"]
            else:
                img_folder = self.data_name["test"]
            data.append(os.path.join(self.root, img_folder, filename + ".jpg"))

        for label in csv.iloc[:, 1:].values:
            targets.append(np.argmax(label))
        targets = np.array(targets)

        return data, targets

    def train_test_split(self, force_override=False):
        split_path = Path(self.root)
        csv_train_path = split_path / "custom_ISIC_2019_Training_GroundTruth.csv"
        csv_test_path = split_path / "custom_ISIC_2019_Test_GroundTruth.csv"

        if force_override:
            if csv_train_path.exists():
                os.remove(csv_train_path)
            if csv_test_path.exists():
                os.remove(csv_test_path)

        if not csv_train_path.exists() or not csv_test_path.exists():
            csv_name = self.csv["full"]
            csv = os.path.join(self.root, csv_name)
            csv = pd.read_csv(csv)
            full_size = len(csv)
            test_size = int(0.25 * full_size)
            rng = np.random.default_rng(12345)
            test_indices = rng.choice(full_size, size=test_size, replace=False)
            test_mask = np.zeros(full_size, dtype=bool)
            test_mask[test_indices] = 1
            csv_test = csv.iloc[test_mask]
            csv_train = csv.iloc[~test_mask]
            csv_train.to_csv(csv_train_path, index=False)
            csv_test.to_csv(csv_test_path, index=False)

        self.csv["train"] = csv_train_path
        self.csv["test"] = csv_test_path
        self.data_name["train"] = self.data_name["full"]
        self.data_name["test"] = self.data_name["full"]

        if force_override:
            self.data, self.targets = self.get_data()
            self.preprocess()


class ISIC2016(AbstractISIC):
    def __init__(
        self,
        root="./data",
        train=True,
        transform=None,
        download=True,
        preprocess=True,
    ):
        """Dataset for ISIC-2016 Dataset.
        Link: https://challenge.isic-archive.com/data/#2016

        Args:
            root (str, optional): Path to root folder. Defaults to "./data".
            train (bool, optional): Use Train or Test Split. Defaults to True.
            transform (Callable[[Image.Image], Image.Image], optional): Pytorch Transform. Defaults to None.
            download (bool, optional): Tries to download dataset. Defaults to True.
            preprocess (bool, optional): Preprocesses dataset for faster readspeed. Defaults to True.
        """
        super().__init__()
        self.folder_name = "ISIC-2016"

        self.csv = {}
        self.csv["test"] = "ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
        self.csv["train"] = "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
        self.data_name = {}
        self.data_name["train"] = "ISBI2016_ISIC_Part3_Training_Data"
        self.data_name["test"] = "ISBI2016_ISIC_Part3_Test_Data"

        self.data_url = {
            "train": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip",
            "test": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip",
        }

        self.csv_url = {
            "train": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv",
            "test": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv",
        }

        self.prep_size = 300  # potentially this could be changed for the test set!
        self.classes_name = ["DIA"]
        self.classes = list(range(len(self.classes_name)))

        self.train = train
        self.root = os.path.join(root, self.folder_name)
        if download:
            self.download()
        self.transform = transform

        self.data, self.targets = self.get_data()
        if preprocess:
            self.preprocess()

    def get_data(self):
        if self.train:
            csv_name = self.csv["train"]
        else:
            csv_name = self.csv["test"]
        csv = os.path.join(self.root, csv_name)
        csvfile = pd.read_csv(csv, header=None)

        raw_data = csvfile.values
        data = []
        targets = []
        for filename, label in raw_data:
            # data.append(os.path.join(self.root, "ISIC2018_Task3_Training_Input", path))
            if self.train:
                img_folder = self.data_name["train"]
            else:
                img_folder = self.data_name["test"]
            data.append(os.path.join(self.root, img_folder, filename + ".jpg"))
            if not isinstance(label, (int, float)):
                label = 0 if label == "benign" else 1
            else:
                label = int(label)
            targets.append(label)
        targets = np.array(targets)

        return data, targets


if __name__ == "__main__":
    import os

    dataroot = os.getenv("DATA_ROOT")

    from tqdm import tqdm

    print("Running Dataset 2016")
    dataset = ISIC2016(root=dataroot, train=False, preprocess=True, download=False)
    print(dataset.targets)

    print(len(dataset))
    from tqdm import tqdm

    for _ in tqdm(dataset):
        pass

    print("Running Dataset 2019")
    dataset = ISIC2019(root=dataroot, train=False, preprocess=True, download=False)
    print(dataset.targets)

    print(len(dataset))
    from tqdm import tqdm

    for _ in tqdm(dataset):
        pass
