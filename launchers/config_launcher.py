import os
from dataclasses import dataclass


@dataclass
class PretrainedArch:
    """DataClass containing the Path to Pretrained Archs.
    Always use add_ckpt method after initialization!"""

    rel_path_to: str  # Path starting from the entrypoint specified
    model_type: str  # Name of Model Type
    dataset: str  # Data on which the model was trained on
    seed: int
    ckpt_path: str = None
    specific_parameters: dict = None

    def add_ckpt(self):
        self.ckpt_path = self.rel_path_to


### Running ###
### IMPLEMENTATION ###
# Add own checkpoints here for loading.

MODEL_LIST = [
    # Pretrained Archs from Cifar10 with ResNets
    PretrainedArch(
        "SSL/cifar10/cifar_resnet18/2022-01-07_16-51-48-043111/checkpoints/last.ckpt",
        "bayesian_resnet",
        "cifar10",
        12345,
    ),
    PretrainedArch(
        "SSL/cifar10/cifar_resnet18/2022-01-07_16-51-48-029128/checkpoints/last.ckpt",
        "bayesian_resnet",
        "cifar10",
        12346,
    ),
    PretrainedArch(
        "SSL/cifar10/cifar_resnet18/2022-01-07_16-51-48-032915/checkpoints/last.ckpt",
        "bayesian_resnet",
        "cifar10",
        12347,
    ),
    # Pretrained Archs from Cifar10 with WideResNets
    PretrainedArch(
        "SSL/cifar10/cifar_wideresnet28-2/2022-01-07_16-51-48-027128/checkpoints/last.ckpt",
        # "wideresnet28-2",
        "bayesian_wide_resnet",
        "cifar10",
        12345,
    ),
    PretrainedArch(
        "SSL/cifar10/cifar_wideresnet28/2022-01-07_16-51-47-999919/checkpoints/last.ckpt",
        # "wideresnet28-2",
        "bayesian_wide_resnet",
        "cifar10",
        12346,
    ),
    PretrainedArch(
        "SSL/cifar10/cifar_wideresnet28/2022-01-07_16-51-47-936850/checkpoints/last.ckpt",
        # "wideresnet28-2",
        "bayesian_wide_resnet",
        "cifar10",
        12347,
    ),
    # Pretrained Archs from Cifar100 with ResNets
    PretrainedArch(
        "SSL/cifar100/cifar_resnet18/2022-01-07_16-51-48-057844/checkpoints/last.ckpt",
        "bayesian_resnet",
        "cifar100",
        12345,
    ),
    PretrainedArch(
        "SSL/cifar100/cifar_resnet18/2022-01-07_16-51-48-061787/checkpoints/last.ckpt",
        "bayesian_resnet",
        "cifar100",
        12346,
    ),
    PretrainedArch(
        "SSL/cifar100/cifar_resnet18/2022-01-07_16-51-48-009660/checkpoints/last.ckpt",
        "bayesian_resnet",
        "cifar100",
        12347,
    ),
    # Pretrained Archs from Cifar100 with WideResNets
    PretrainedArch(
        "SSL/cifar100/cifar_wideresnet28-2/2022-01-07_16-51-48-008666/checkpoints/last.ckpt",
        # "wideresnet28-2",
        "bayesian_wide_resnet",
        "cifar100",
        12345,
    ),
    PretrainedArch(
        "SSL/cifar100/cifar_wideresnet28-2/2022-01-07_16-51-47-988751/checkpoints/last.ckpt",
        # "wideresnet28-2",
        "bayesian_wide_resnet",
        "cifar100",
        12346,
    ),
    PretrainedArch(
        "SSL/cifar100/cifar_wideresnet28-2/2022-01-07_16-51-47-990133/checkpoints/last.ckpt",
        # "wideresnet28-2",
        "bayesian_wide_resnet",
        "cifar100",
        12347,
    ),
]


def get_pretrained_arch(dataset: str, model: str, seed: int) -> PretrainedArch:
    """
    Retrieves a pretrained architecture from a global list based on the specified dataset,
    model type, and seed.

    Args:
        dataset (str): The name of the dataset.
        model (str): The type of the model architecture.
        seed (int): The seed value associated with the pretrained architecture.

    Returns:
        PretrainedArch: The pretrained architecture matching the provided specifications.

    Raises:
        NotImplementedError: If no pretrained architecture matches the specified dataset,
            model type, and seed in the global MODEL_LIST.

    Note:
        This function searches the global MODEL_LIST for a pretrained architecture that
        matches the specified dataset, model type, and seed. If found, it returns the
        corresponding pretrained architecture. If no match is found, a NotImplementedError
        is raised with a message indicating the absence of a pretrained architecture with
        the provided specifications.

    Example:
        If called with ('CIFAR-10', 'ResNet', 42), the function searches for a pretrained
        architecture in MODEL_LIST with dataset='CIFAR-10', model_type='ResNet', and seed=42.
        If a match is found, the corresponding PretrainedArch object is returned.
    """
    for pretrained_arch in MODEL_LIST:
        if (
            model == pretrained_arch.model_type
            and dataset == pretrained_arch.dataset
            and seed == pretrained_arch.seed
        ):
            return pretrained_arch

    raise NotImplementedError(
        "There is no pretrained arch with these specifics in global MODEL_LIST: \n Dataset:{} \n Model:{} \n Seed: {}".format(
            dataset, model, seed
        )
    )


for model in MODEL_LIST:
    model.add_ckpt()


if __name__ == "__main__":
    print("Length of Model List: {}".format(len(MODEL_LIST)))
    dataset = "cifar10"
    # model = "wideresnet28-2"
    model = "bayesian_wide_resnet"
    seed = 12345
    pretrained_arch = get_pretrained_arch(dataset, model, seed)
    assert (
        pretrained_arch.ckpt_path
        == "SSL/cifar10/cifar_wideresnet28-2/2022-01-07_16-51-48-043111/checkpoints/last.ckpt"
    )

    dataset = "cifar10"
    model = "wideresnet28"
    try:
        pretrained_arch = get_pretrained_arch(dataset, model, seed)
    except NotImplementedError as E:
        print(E)
        print("Succesfull Run")
