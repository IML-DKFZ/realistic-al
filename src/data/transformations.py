from typing import List

import torch
from torchvision import transforms
from torchvision.transforms import Lambda, RandAugment

from .randaugment import CutoutAbs, RandAugmentMC, RandAugmentMCCutout, RandAugmentPC
from .randaugment_2 import RandAugmentMC_FixMatch


def get_transform(
    name: str = "basic",
    mean: List[int] = [0],
    std: List[int] = [1],
    shape: List[int] = None,
):
    """Transformation selection function.

    Args:
        name (str, optional): Name of transformation. Defaults to "basic".
        mean (List[int], optional): Mean for normalization. Defaults to [0].
        std (List[int], optional): STD for formalization. Defaults to [1].
        shape (List[int], optional): Desired shape -- currently unused. Defaults to None.

    Returns:
        Transform: For image Data (IMG->IMG) or tabular (NPY->NPY)
    """
    ### IMPLEMENTATION ###
    # New transformations can be added here.
    transform = []
    if name == "cifar_basic":
        transform.append(_get_baseline_cifar_transfrom())
    elif name == "cifar_weak":
        transform.append(_get_weak_cifar_transform())
    elif name == "cifar_randaugment":
        transform.append(_get_randaug_cifar_transform())
    elif name == "cifar_randaugment_cutout":
        transform.append(_get_randaug_cifar_cutout_transform())
    elif name == "cifar_randaugmentMC":
        return _get_cifar_rand(mean, std)
    elif name == "isic_train":
        transform.append(_get_isic_train_transform())
    elif name == "isic_randaugment":
        transform.append(_get_isic_randaug_transform())
    elif name == "isic_randaugmentMC":
        transform.append(_get_isic_randaugMC())
    elif name == "isic_randaugtensor":
        return get_isic_randaug_trafo(mean=mean, std=std)
    elif name == "resize_224":
        transform.append(_resize_transform(224))
    elif name == "toy_gauss_0.05":
        return ToyNoiseTransform(sig=0.05)
    elif name == "toy_identity":
        return IdentityTransform()
    elif name == "imagenet_train":
        transform.append(_get_imagenet_train_transform())
    elif name == "imagenet_randaug":
        transform.append(_get_imagenet_randaug_transform())
    elif name == "imagenet_randaug_cutout":
        transform.append(_get_imagenet_randaug_cutout_transform())
    elif name == "imagenet_randaugMC":
        transform.append(_get_imagenet_randaugMC())
    elif name == "imagenet_test":
        transform.append(_get_imagenet_test_transform())

    transform.append(_get_norm_transform(mean, std))
    transform = transforms.Compose(transform)
    return transform


def _get_imagenet_train_transform():
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    )
    return transform_train


def _get_cifar_rand(mean, std):
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            RandAugmentMCCutout(n=2, m=10, prob=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_train


def _get_imagenet_test_transform():
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    return transform


def _get_imagenet_randaug_transform():
    transform = transforms.Compose(
        [_get_imagenet_train_transform(), RandAugmentPC(n=1, m=2, cut_rel=0)]
    )
    return transform


def _get_imagenet_randaugMC():
    transform = transforms.Compose(
        [_get_imagenet_train_transform(), RandAugmentMC(n=2, m=10, cut_rel=0, prob=1)]
    )
    return transform


def _get_imagenet_randaug_cutout_transform():
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            # transforms.RandomErasing()
            Lambda(lambda x: CutoutAbs(x, int(x.size[0] * 0.2))),
        ]
    )
    return transform


def _get_norm_transform(mean, std):
    norm_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return norm_transform


def _get_baseline_cifar_transfrom():
    baseline_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=32, padding=4)]
    )
    return baseline_transform


def _get_weak_cifar_transform():
    weak_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
        ]
    )
    return weak_transform


def _get_randaug_cifar_transform():
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


def _get_isic_train_transform():
    """Return Transformation for ISIC Skin Lesion Diagnosis.
    Based on: https://github.com/JiaxinZhuang/Skin-Lesion-Recognition.Pytorch
    """
    re_size = 300
    input_size = 224
    train_transform = transforms.Compose(
        [
            # transforms.Resize(re_size), # resize is not necessary if data is preprocessed!
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine(
                [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]
            ),
            transforms.RandomCrop(input_size),
        ]
    )
    return train_transform


def _get_isic_randaugMC():
    re_size = 300
    input_size = 224
    train_transform = transforms.Compose(
        [
            # transforms.Resize(re_size), # resize is not necessary if data is preprocessed!
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01), put this in a later stage!
            # transforms.RandomRotation([-180, 180]), # Affine already rotates!
            transforms.RandomAffine(
                [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]
            ),
            transforms.RandomCrop(input_size),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            RandAugmentMC(n=2, m=10, prob=1, cut_rel=0),
        ]
    )
    return train_transform


def _get_isic_randaug_transform():
    re_size = 300
    input_size = 224
    train_transform = transforms.Compose(
        [
            # transforms.Resize(re_size), # resize is not necessary if data is preprocessed!
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine(
                [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]
            ),
            RandAugmentMC(n=1, m=2),
            transforms.RandomCrop(input_size),
        ]
    )
    return train_transform


def get_isic_randaug_trafo(mean, std):
    re_size = 300
    input_size = 224
    train_transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            # transforms.Resize(re_size), # resize is not necessary if data is preprocessed!
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine(
                [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]
            ),
            RandAugmentMC_FixMatch(num_ops=1, max_magnitude=2),
            transforms.RandomCrop(input_size),
            transforms.Lambda(lambda x: x.to(dtype=torch.float) / 255),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform


def _resize_transform(input_size=224):
    # input_size = 224
    transform = transforms.Compose([transforms.Resize((input_size, input_size))])
    return transform


def _get_randaug_cifar_cutout_transform():
    randaug_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            RandAugment(),
            Lambda(lambda x: CutoutAbs(x, int(x.size[0] * 0.25))),
            # RandAugmentMCCutout(n=1, m=2),
        ]
    )
    return randaug_transform


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
