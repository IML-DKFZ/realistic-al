import os

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms


class ISIC2016(Dataset):
    """SKin Lesion"""

    def __init__(
        self,
        root="./data",
        train=True,
        transform=None,
        download=False,
        preprocess=True,
    ):
        self.transform = transform
        self.train = train
        self.root = os.path.join(root, "ISIC-2016")

        self.data, self.targets = self.get_data(self.root)
        # self.classes_name = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        self.classes_name = ["DIA"]
        self.classes = list(range(len(self.classes_name)))
        # self.target_img_dict = {}
        # targets = np.array(self.targets)
        # for target in self.classes:
        #     indexes = np.nonzero(targets == target)[0]
        #     self.target_img_dict.update({target: indexes})
        if preprocess:
            self.preprocess()

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

    def __len__(self):
        return len(self.data)

    def get_data(self, data_dir):

        # if self.train:
        #     csv = "split_data/split_data_{}_fold_train.csv".format(iterNo)
        # else:
        #     csv = "split_data/split_data_{}_fold_test.csv".format(iterNo)

        if self.train:
            csv = os.path.join(
                self.root, "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
            )
        else:
            csv = os.path.join(self.root, "ISBI2016_ISIC_Part3_Test_GroundTruth.csv")

        fn = os.path.join(data_dir, csv)
        csvfile = pd.read_csv(fn)

        raw_data = csvfile.values

        data = []
        targets = []
        for filename, label in raw_data:
            # data.append(os.path.join(self.root, "ISIC2018_Task3_Training_Input", path))
            if self.train:
                data.append(
                    os.path.join(
                        self.root,
                        "ISBI2016_ISIC_Part3_Training_Data",
                        filename + ".jpg",
                    )
                )
            else:
                data.append(
                    os.path.join(
                        self.root, "ISBI2016_ISIC_Part3_Test_Data", filename + ".jpg"
                    )
                )
            label = 0 if label == "benign" else 1
            targets.append(label)
        targets = np.array(targets)

        return data, targets

    def preprocess(self):
        from pathlib import Path

        basepath = Path(self.root)
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
                img = img.resize((300, 300), resample=Image.BILINEAR)
                if not preprocessed_path.parent.exists():
                    preprocessed_path.parent.mkdir()

                img.save(preprocessed_path)
            preprocessed_data.append(preprocessed_path)
        self.data = preprocessed_data

    def download(self):
        if self.train:
            train_gt = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
            train_data = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip"
            pass
        else:
            pass
            test_gt = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
            test_data = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip"


# def default_loader(path):
#     from torchvision import get_image_backend

#     if get_image_backend() == "accimage":
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)


# def pil_loader(path):
#     """Image Loader
#     """
#     with open(path, "rb") as afile:
#         img = Image.open(afile)
#         return img.convert("RGB")


# def accimage_loader(path):
#     import accimage

#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def print_dataset(dataset, print_time):
    print(len(dataset))
    from collections import Counter

    counter = Counter()
    labels = []
    for index, (img, label) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label)
        labels.append(label)
    counter.update(labels)
    print(counter)


if __name__ == "__main__":
    import os

    dataroot = os.getenv("DATA_ROOT")
    dataset = ISIC2016(root=dataroot, train=True, preprocess=False)

    print(len(dataset))
    from tqdm import tqdm

    for _ in tqdm(dataset):
        pass

    dataset = ISIC2016(root=dataroot, train=True, preprocess=True)

    print(len(dataset))
    from tqdm import tqdm

    for _ in tqdm(dataset):
        pass


# re_size = 300
# input_size = 224
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# train_transform = transforms.Compose(
#     [
#         transforms.Resize(re_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
#         transforms.RandomRotation([-180, 180]),
#         transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
#         transforms.RandomCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ]
# )

# val_transform = transforms.Compose(
#     [
#         transforms.Resize((input_size, input_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ]
# )

