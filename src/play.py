import os
from data.skin_dataset import ISIC2019
from tqdm import tqdm


dataset = ISIC2019(
    root=os.getenv("DATA_ROOT"), train=True, preprocess=False, download=True
)
dataset_prep = ISIC2019(root=os.getenv("DATA_ROOT"), train=True, preprocess=True)

if __name__ == "__main__":
    print(len(dataset))
    # for x, y in tqdm(dataset):
    #     pass

    for x, y in tqdm(dataset_prep):
        pass
