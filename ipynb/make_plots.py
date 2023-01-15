import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotlib.performance_plots import plot_standard_dev
from utils.file_utils import get_experiment_df


sns.set_style("whitegrid")


def dataset_name_to_id(dataset_name: str) -> tuple[str, str]:
    """Convert human level dataset name to computer-readable dataset identifier. Also, select fill set and check if it is a subset.

    Args:
        dataset_name: human-level name of the dataset

    Returns:
        normalized dataset and fillset identifiers
    """

    datasets = {
        "CIFAR-10": "cifar10",
        "CIFAR-100": "cifar100",
        "CIFAR-10-LT": "cifar10_imb",
        "MIO-TCD": "miotcd",
        "ISIC-2019": "isic2019",
    }

    sub_datasets = {"CIFAR-10-LT": datasets["CIFAR-10"], "ISIC-2019": "isic19"}

    dataset_id = datasets[dataset_name]
    fillset_id = sub_datasets.get(dataset_name, dataset_id)

    return dataset_id, fillset_id


def build_setting_path_mapping(
    dataset_id: str, fillset_id: str, base_path: Path
) -> dict[str, list[Path]]:
    """Create a dict that maps experiment settings to data paths.

    Args:
        dataset_id: normalized dataset name for this group of experiments
        fillset_id: normalized fillset name for this group of experiments
        base_path: path where all experiments are saved

    Returns:
        a mapping from experiment settings to folders of experiment data
    """
    settings = {
        "full": ["low", "med", "high"],
        "low-bb": ["low-batchbald"],
        "low-bb-fulll": ["low-batchbald", "low"],
        "bblow": ["low"],
    }

    for setting in settings:
        settings[setting] = [f"active-{fillset_id}_{exp}" for exp in settings[setting]]

    setting_paths: dict[str, list[Path]] = {}

    for setting, experiments in settings.items():
        setting_paths[setting] = [base_path / dataset_id / exp for exp in experiments]

    return setting_paths


# %%
savepath = Path("./plots").resolve()
dataset = "CIFAR-100"

full_models = {
    "CIFAR-10": {
        "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.0005_lr-0.1_optim-sgd_cosine",
        "PT": None,
    },
    "CIFAR-100": {
        "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.005_lr-0.1_optim-sgd_cosine",
        "PT": None,
    },
    "CIFAR-10-LT": {
        "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.0005_lr-0.1_optim-sgd_cosine_weighted-true",
        "PT": None,
    },
    "MIO-TCD": {
        "Basic": "basic_model-resnet_drop-0_aug-imagenet_randaugMC_wd-5e-05_lr-0.1_optim-sgd_cosine_weighted-True_epochs-80",
        "PT": None,
    },
    "ISIC-2019": {
        "Basic": "basic_model-resnet_drop-0_aug-isic_train_wd-0.005_lr-0.01_optim-sgd_cosine_weighted-True_epochs-200",
        "PT": None,
    },
}

base_path = Path(
    "~/NetworkDrives/E130-Personal/Lüth/carsten_al_cvpr_2023-November/logs_cluster/activelearning/"
).expanduser()


d_set, fill_set = dataset_name_to_id(dataset)
setting_paths = build_setting_path_mapping(d_set, fill_set, base_path)


# %%

# print(dataset)
full_paths = {}
for model in full_models[dataset]:
    if full_models[dataset][model] is not None:
        full_paths[model] = (
            base_path / d_set / "full_data" / full_models[dataset][model]
        )

from pprint import pprint

pprint(setting_paths)
print(full_paths)

# %%
match_patterns = [
    r"basic_.*",
    r"basic-pretrained_.*",
    #     r".*__wloss.*"
    #     BB experiment
    #     r".*bald.*"
    #     r".*random.*"
    r"fixmatch_.*",
    #     r"fixmatch-pretrained_.*",
]

filter_dict = {"standard": [".*batchbald.*"], "bb": []}
# filter_patterns = [

# #     r".*batchbald.*"
# #     ".*wd-0.01_.*"
# #     r".*kcenter.*",
# #     r".*variationratios.*"
# #     r".*batchbald.*"
# #     r".*basic-pretrained_.*",
# ]

# %%
hue_name = "Acquisition"

hue_split = "acq-"

style_name = "PreTraining & Semi-Supervised"
style_fct = lambda x: "PT: {}, Sem-SL: {}".format(
    "pretrained_model" in x.name, "fixmatch" in x.name
)

unit_vals = None
unit_name = "Unit"

palette = {
    "bald": "tab:blue",
    "kcentergreedy": "tab:green",
    "entropy": "tab:orange",
    "random": "tab:red",
    "batchbald": "tab:cyan",
}

# Sadly does not Work!
# dashes = {
#     'PT: False, Sem-SL: False' : '--',
#     'PT: True, Sem-SL: False' : '-',
#     'PT: False, Sem-SL: True' : ':',
#     'PT: True, Sem-SL: True' : '-.',
# }

dashes = {
    "PT: False, Sem-SL: False": (4, 4),
    "PT: True, Sem-SL: False": (1, 0),
    "PT: False, Sem-SL: True": (1, 2),
    "PT: True, Sem-SL: True": (2, 1),
}

markers = True
# no color
# markers = {
#     'PT: False, Sem-SL: False' : "x",
#     'PT: True, Sem-SL: False' : "+",
#     'PT: False, Sem-SL: True' : "1",
#     'PT: True, Sem-SL: True' : "2",
# }

markers = {
    "PT: False, Sem-SL: False": "v",
    "PT: True, Sem-SL: False": "o",
    "PT: False, Sem-SL: True": "D",
    "PT: True, Sem-SL: True": "X",
}

err_kws = {"alpha": 0.3}

# %%
training_settings = {
    "all": [
        "PT: False, Sem-SL: False",
        "PT: True, Sem-SL: False",
        "PT: False, Sem-SL: True",
        "PT: True, Sem-SL: True",
    ],
    "basic": ["PT: False, Sem-SL: False"],
    "Self-SL": ["PT: True, Sem-SL: False"],
    "Sem-SL": ["PT: False, Sem-SL: True"],
    "Self-Sem-SL": ["PT: True, Sem-SL: True"],
}

# %%
setting_dfs = dict()
for setting in setting_paths:
    base_paths = setting_paths[setting]
    if not all(base_path.is_dir() for base_path in base_paths):
        print(f"Skipping Setting: {setting}\nPath is not existent {base_path}")
        continue
    key_filter = "standard"
    for key in filter_dict:
        if key in setting:
            key_filter = key
    if key is None:
        key_filter = "standard"
    print("Selecting Filter Pattern from {}".format(key_filter))
    filter_patterns = filter_dict[key_filter]
    dfs = []
    #     print(base_paths)

    for base_path in base_paths:
        paths = [path for path in base_path.iterdir() if path.is_dir()]
        paths.sort()
        print("Folders in Path: \n {}\n".format(base_path))

        experiment_paths = []
        for path in paths:
            #         print(path)
            for pattern in match_patterns:
                #             print(path.name)
                out = re.match(pattern, str(path.name))
                if out is not None:
                    print(path.name)
                    skip = False
                    for filter_pattern in filter_patterns:
                        if re.match(filter_pattern, str(path)) is not None:
                            skip = True
                    if skip:
                        continue

                    print(path.name)
                    experiment_paths.append(path)
                    continue

        hue_names = [
            path.name.split(hue_split)[1].split("_")[0] for path in experiment_paths
        ]  # .split('_')[0] for path in paths]
        style_vals = [style_fct(path) for path in experiment_paths]

        df = []
        for i, (base_dir) in enumerate(experiment_paths):
            base_dir = Path(base_dir)
            if hue_names is not None:
                hue_val = hue_names[i]
            else:
                hue_val = None
            if style_vals is not None:
                style_val = style_vals[i]
            else:
                style_val = None
            if unit_vals is not None:
                unit_val = unit_vals[i]
            else:
                unit_val = None

            experiment_frame = get_experiment_df(base_dir, name=hue_val)
            # experiment_frame[hue_name] = hue_val
            if experiment_frame is None:
                continue

            # Add new metric values
            experiment_add = get_experiment_df(
                base_dir, pattern="test_metrics.csv", name=hue_val
            )
            #         print(experiment_add)
            if experiment_add is not None:
                #             print(experiment_frame)
                #             print(experiment_add)
                del experiment_add["Name"]
                del experiment_add["version"]
                experiment_frame = experiment_frame.join(experiment_add)
            #             print(experiment_frame)

            experiment_frame[hue_name] = hue_val
            experiment_frame[style_name] = style_val
            experiment_frame[unit_name] = unit_val
            df.append(experiment_frame)
        df = pd.concat(df)
        df.reset_index(inplace=True)

        dfs.append(df)
        setting_dfs[setting] = dfs

# %%
full_data_dict = {}

for key, path in full_paths.items():
    test_acc_df = pd.read_csv(path / "test_acc.csv", index_col=0)
    full_data_dict[key] = dict()
    full_data_dict[key]["test_acc"] = dict()
    full_data_dict[key]["test_acc"]["mean"] = float(test_acc_df["Mean"])
    full_data_dict[key]["test_acc"]["std"] = float(test_acc_df["STD"])

    if (path / "test_w_acc.csv").is_file():
        mean_recall_df = pd.read_csv(path / "test_w_acc.csv", index_col=0)
        full_data_dict[key]["test/w_acc"] = dict()
        full_data_dict[key]["test/w_acc"]["mean"] = float(mean_recall_df["Mean"])
        full_data_dict[key]["test/w_acc"]["std"] = float(mean_recall_df["STD"])

# %%
from itertools import product

plot_values = {
    "Accuracy": "test_acc",
    "Balanced Accuracy": "test/w_acc",
    "Batch Entropy": "Acquisition Entropy",
    "Acquired Entropy": "Dataset Entropy",
}
upper_bounds = [True, False]
y_shareds = [True, False]


for y_shared, upper_bound in product(upper_bounds, y_shareds):
    for plot_val, plot_key in plot_values.items():
        if plot_val in ["Batch Entropy", "Acquired Entropy"]:
            if not (y_shared and upper_bound):
                continue

        for setting in setting_dfs:
            dfs = setting_dfs[setting]
            if plot_key in dfs[0]:
                for training_setting, training_styles in training_settings.items():
                    num_cols = len(dfs)
                    ax_legend = 0
                    fig, axs = plt.subplots(ncols=num_cols, sharey=y_shared)
                    if num_cols == 1:
                        axs = [axs]
                    if style_vals is None:
                        style_name = None
                    if unit_vals is None:
                        unit_name = None

                    for i in range(num_cols):
                        df = dfs[i][dfs[i][style_name].isin(training_styles)]
                        if len(df) == 0:
                            continue
                        ax = axs[i]

                        legend = False
                        if i == ax_legend:
                            legend = "auto"
                        ax = plot_standard_dev(
                            ax,
                            df,
                            y=plot_key,
                            hue=hue_name,
                            style=style_name,
                            units=unit_name,
                            ci="sd",
                            legend=legend,
                            palette=palette,
                            markers=markers,
                            dashes=dashes,
                            err_kws={"alpha": 0.2},
                        )  # , units=unit_name)
                        full_dict = {
                            "Basic": {"color": "black", "linestyle": "--"},
                            "PT": {"color": "black", "linestyle": "- "},
                        }
                        if upper_bound:
                            for model in full_dict:
                                if model in full_data_dict:
                                    if plot_key in full_data_dict[model]:
                                        ax.axhline(
                                            full_data_dict[model][plot_key]["mean"]
                                            / 100,
                                            **full_dict[model],
                                        )
                        ax.set_xlabel("Labeled Samples")
                        ax.set_ylabel(plot_val)
                        if i == ax_legend:
                            ax.get_legend().remove()
                    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                    #           ncol=3, fancybox=True, shadow=True)
                    # fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    if num_cols == 3:
                        ncol_legend = 5
                    else:
                        ncol_legend = 3
                    fig.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.05),
                        fancybox=True,
                        shadow=True,
                        ncol=ncol_legend,
                    )

                    fig.set_size_inches(4 * num_cols, 6)
                    fig.tight_layout()
                    save_dir = savepath / dataset / plot_val.replace(" ", "-")
                    if not save_dir.is_dir():
                        save_dir.mkdir(parents=True)
                    if upper_bound is True and y_shared is True:
                        fn = f"plot-{setting}_train-{training_setting}.pdf"
                    else:
                        fn = f"plot-{setting}_train-{training_setting}_yshared-{y_shared}_bound-{upper_bound}.pdf"
                    print(f"Filename:{fn}")
                    plt.savefig(save_dir / fn, bbox_inches="tight")
                    # plt.show()

# %%

# %%
dfs[i][style_name].isin(hue_settings)

# %% [markdown]
# ## Path Selection

# %%
dfs[0]

# %%
from IPython.display import HTML

# %%
val_df = dfs[0].loc[
    :, [col for col in dfs[0].columns if "test/rec" in col or col == "num_samples"]
]
val_df
# dfs[0].columns[0]
HTML(val_df.groupby("num_samples").mean().to_html(classes="table table-stripped"))

# %%
test = [x for x in range(10) if x % 2 == 0]
test

# %% [markdown]
# ## Obtain Numerical Results

# %%
metric = "test_acc"

# %%
df = dfs[0]
df_random = df[df["Name"] == "random"]
# df_random = df.loc[:, ["test_acc", "num_samples", "PreTraining & Semi-Supervised"]]

# %%
for i in range(len(dfs)):
    df = dfs[i]
    df_random = df[df["Name"] == "random"]
    df_random = (
        df_random.groupby(
            ["num_samples", "PreTraining & Semi-Supervised"],
        )
        .agg({metric: ["mean", "std"]})
        .round(4)
        * 100
    )
    print(base_paths[i])
    print(df_random)

# %%

# %%
