import torch
from torchvision import transforms

from .randaugment import RandAugmentMC


def get_transform(name="basic", mean=[0], std=[1], shape=None):
    transform = []
    if name == "cifar_basic":
        transform.append(get_baseline_cifar_transfrom())
    elif name == "cifar_weak":
        transform.append(get_weak_cifar_transform())
    elif name == "cifar_randaugment":
        transform.append(get_randaug_cifar_transform())
    # TODO: Make this nice down the line -- see how other people do stuff like this!
    elif name == "isic_train":
        transform.append(get_isic_train_transform())
    elif name == "resize_224":
        transform.append(resize_transform(224))
    elif name == "toy_gauss_0.05":
        return ToyNoiseTransform(sig=0.05)
    elif name == "toy_identity":
        return IdentityTransform()

    transform.append(get_norm_transform(mean, std))
    transform = transforms.Compose(transform)
    return transform


def get_norm_transform(mean, std):
    norm_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return norm_transform


def get_baseline_cifar_transfrom():
    baseline_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=32, padding=4)]
    )
    return baseline_transform


def get_weak_cifar_transform():
    weak_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
        ]
    )
    return weak_transform


def get_randaug_cifar_transform():
    """best transformation for wideresnet 28-2 on cifar10 according to:
    https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf
    """
    randaug_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            RandAugmentMC(n=1, m=2),
        ]
    )
    return randaug_transform


def get_isic_train_transform():
    """Return Transformation for ISIC Skin Lesion Diagnosis.
    Based on: https://github.com/JiaxinZhuang/Skin-Lesion-Recognition.Pytorch
    """
    re_size = 300
    input_size = 224
    train_transform = transforms.Compose(
        [
            transforms.Resize(re_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine(
                [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]
            ),
            transforms.RandomCrop(input_size),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform


def resize_transform(input_size=224):
    # input_size = 224
    transform = transforms.Compose([transforms.Resize((224, 224))])
    return transform


class AbstractTransform:
    def __call__(self, x):
        pass

    def __repr__(self):
        ret_str = (
            str(type(self).__name__)
            + "( "
            + ", ".join([key + " = " + repr(val) for key, val in self.__dict__.items()])
            + " )"
        )
        return ret_str


class IdentityTransform(AbstractTransform):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


class ToyNoiseTransform(AbstractTransform):
    def __init__(self, sig=0.05):
        self.sig = sig

    def __call__(self, x: torch.Tensor):
        out = x + torch.randn_like(x) * self.sig
        return out
