import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style()


def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


def visualize_samples(vis_data: np.ndarray, acq_vals):
    num_samples = len(vis_data)
    n_rows, n_cols = 8, 8
    if len(vis_data.shape) == 3:
        vis_data = vis_data[:, None, :, :]
    assert len(vis_data.shape) == 4
    if vis_data.shape[1] == 1:
        vis_data = np.concatenate([vis_data] * 3, axis=1)
    vis_data = vis_data.transpose(0, 2, 3, 1)
    fig, ax = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            ind = i * 8 + j
            ax[i][j].set_axis_off()
            if ind >= num_samples:
                continue
            ax[i][j].imshow(vis_data[ind])
            ax[i][j].set_title("{:.3f}".format(acq_vals[ind]))
    fig.tight_layout()
    return fig, ax


def visualize_labels(acq_labels, num_classes):
    fig, ax = plt.subplots()
    values, counts = np.unique(acq_labels, return_counts=True)
    counts = counts / counts.sum()
    ax.bar(x=values, height=counts)
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_xticks(np.arange(num_classes))
    return fig, ax
