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
