import umap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

param_combinations = [
    {"n_neighbors": 20, "min_dist": 1, "metric": "euclidean", "low_memory": True},
    {"n_neighbors": 15, "min_dist": 0.01, "metric": "cosine", "low_memory": True},
    {"n_neighbors": 30, "min_dist": 0.2, "metric": "correlation", "low_memory": True},
    {"n_neighbors": 50, "min_dist": 0.5, "metric": "manhattan", "low_memory": True},
]

def plot_umap(test_path, metric: int, s):
    test_path = Path(test_path)
    features = np.load(test_path / 'test_x.npy')
    lable_gt = np.load(test_path / 'test_y.npy')
    lable_out = np.load(test_path / f'output_y.npy')

    reducer = umap.UMAP(**param_combinations[metric])
    X_umap = reducer.fit_transform(features)

    accuracy = ((lable_gt == lable_out)).sum() / lable_gt.shape[0]

    unique_labels_gt = np.unique(lable_gt)
    unique_labels_out = np.unique(lable_out)

    # Define a gradient of orange shades and reserve the last one for grey
    num_colors = len(unique_labels_gt) - 1  # All but last label
    orange_cmap = plt.get_cmap("Oranges")

    if num_colors > 1:
        orange_shades = [orange_cmap(i / (num_colors - 1)) for i in range(num_colors)]
    else:
        orange_shades = [orange_cmap(0.5)]  # Single orange shade if only one label

    colors = orange_shades + ["#808080"]  # Grey for last label

    color_dict = {label: colors[i] for i, label in enumerate(unique_labels_gt)}

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"{test_path.name}")

    # Ground truth plot
    scatter = axs[0].scatter(X_umap[:, 0], X_umap[:, 1], c=[color_dict[label] for label in lable_gt], s=s, alpha=0.6)
    axs[0].set_title("Ground truth UMAP")
    axs[0].set_xlabel("UMAP Dimension 1")
    axs[0].set_ylabel("UMAP Dimension 2")

    # Output plot
    scatter = axs[1].scatter(X_umap[:, 0], X_umap[:, 1], c=[color_dict[label] for label in lable_out], s=s, alpha=0.6)
    axs[1].set_title(f"Output UMAP acc ({accuracy.round(3)})")
    axs[1].set_xlabel("UMAP Dimension 1")
    axs[1].set_ylabel("UMAP Dimension 2")

    plt.show()
    plt.close()

