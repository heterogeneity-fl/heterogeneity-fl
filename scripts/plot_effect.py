""" Plot training results. """

import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


METRICS = ["train_losses", "test_accuracies"]
ALGS = ["episode_mem", "local_clip", "naive_parallel_clip", "minibatch_clip", "scaffold_clip", "episode"]
BEST = {
    "train_losses": lambda vals: np.min(vals),
    "test_accuracies": lambda vals: np.max(vals),
}
NAMES = {
    "train_losses": "Train Loss",
    "test_accuracies": "Test Accuracy",
    "episode_mem": "EPISODE++",
    "local_clip": "CELGC",
    "naive_parallel_clip": "NaiveParallelClip",
    "minibatch_clip": "Clipped Minibatch SGD",
    "scaffold_clip": "SCAFFOLDClip",
    "episode": "EPISODE",
    "S": "Participating Clients",
    "H": "Data similarity",
}
EFFECT_VAL_TRANSFORM = {
    "S": lambda S: S,
    "H": lambda H: f"{round(100*(1-float(H)))}%"
}
PLOT_SIZE = 4
LINE_WIDTH = 2.5
MARKERS = ["o", "v", "^", "s", "P", "X"]
MARKER_SIZE = 8
HORIZONTAL_STRETCH = 1.0


plt.rcParams.update({'font.size': 16})


def read_single_effect(family_dir):
    """ Read results for each value of a single effect variable and each algorithm. """

    # Read results for each value of effect variable and each algorithm.
    effect_name = None
    effect_vals = []
    results = {}
    i = 0
    while True:

        # Read results for next value of effect variable.
        effect_dirs = glob.glob(os.path.join(family_dir, f"{i}_*"))
        if len(effect_dirs) > 1:
            raise ValueError("Invalid results directory formatting.")
        if len(effect_dirs) == 0:
            break
        effect_dir = effect_dirs[0]

        # Get effect name.
        name = os.path.basename(effect_dir)
        und_pos_1 = name.find("_")
        und_pos_2 = name.find("_", und_pos_1 + 1)
        assert und_pos_1 > 0 and und_pos_2 > 0
        current_name = name[und_pos_1+1: und_pos_2]
        if effect_name is None:
            effect_name = current_name
        else:
            assert effect_name == current_name

        # Get value of effect variable.
        effect_val = name[und_pos_2+1:]
        effect_vals.append(effect_val)
        results[effect_val] = {}

        # Read results of each algorithm for this effect value.
        for alg in ALGS:
            alg_files = glob.glob(os.path.join(effect_dir, alg, "*.json"))
            candidate_avg = [filename for filename in alg_files if "Rank" not in filename]
            if len(candidate_avg) != 1:
                raise ValueError(f"Incomplete results for {effect_dir}/{alg}.")
            avg_filename = candidate_avg[0]
            with open(avg_filename, "r") as f:
                results[effect_val][alg] = json.load(f)

        i += 1

    if i == 0:
        raise ValueError(f"No results found under {family_dir}.")

    return effect_name, effect_vals, results


def make_subplot(plot_idx, nplots, ax, effect_name, metric, results, effect_vals, name):
    """ Create a single subplot. """

    # Plot results.
    x = np.array([EFFECT_VAL_TRANSFORM[effect_name](effect_val) for effect_val in effect_vals])
    y_mins = []
    y_maxs = []
    for i, alg in enumerate(ALGS):

        y = np.array([
            BEST[metric](results[effect_val][alg][metric])
            for effect_val in effect_vals
        ])
        plot_kwargs = {
            "linewidth": LINE_WIDTH,
            "marker": MARKERS[i],
            "markersize": MARKER_SIZE,
        }
        if plot_idx == 0:
            plot_kwargs["label"] = NAMES[alg]
        ax.plot(x, y, **plot_kwargs)

        y_mins.append(float(np.min(y)))
        y_maxs.append(float(np.max(y)))

    ax.set_xlabel(NAMES[effect_name])
    ax.set_ylabel(NAMES[metric])
    """
    if plot_idx == (nplots - 1) // 2:
        ax.set_title(name, pad=8.0)
    """
    y_min = np.min(y_mins)
    y_max = np.max(y_maxs)

    return y_min, y_max


def plot(family_dirs, name):

    # Read results for all directories.
    effect_vars = []
    total_results = {}
    total_effect_vals = {}
    for family_dir in family_dirs:
        effect_name, effect_vals, results = read_single_effect(family_dir)
        effect_vars.append(effect_name)
        total_results[effect_name] = dict(results)
        total_effect_vals[effect_name] = list(effect_vals)

    # Make subplots.
    nplots = len(family_dirs) * len(METRICS)
    fig, axs = plt.subplots(nrows=1, ncols=nplots)
    fig.set_size_inches(
        nplots * PLOT_SIZE * HORIZONTAL_STRETCH, PLOT_SIZE
    )
    i = 0
    metric_y_min = {}
    metric_y_max = {}
    for effect_var in effect_vars:
        for metric in METRICS:
            y_min, y_max = make_subplot(
                i,
                nplots,
                axs[i],
                effect_var,
                metric,
                total_results[effect_var],
                total_effect_vals[effect_var],
                name,
            )

            if metric not in metric_y_min:
                metric_y_min[metric] = []
                metric_y_max[metric] = []
            metric_y_min[metric].append(y_min)
            metric_y_max[metric].append(y_max)

            i += 1

    # Match axes of plots with the same metric.
    for i, metric in enumerate(METRICS):
        y_min = np.min(metric_y_min[metric])
        y_max = np.max(metric_y_max[metric])
        y_min -= (y_max - y_min) * 0.05
        y_max += (y_max - y_min) * 0.05
        for j in range(nplots):
            if j % len(effect_vars) == i:
                axs[j].set_ylim([y_min, y_max])

    # Set legend and layout.
    fig.suptitle(name, y=0.9)
    #plt.subplots_adjust(top=1.5)
    plt.figlegend(loc="lower center", ncol=len(ALGS), bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()

    # Save plot to file.
    parent = os.path.normpath(os.path.join(family_dirs[0], ".."))
    if name is None:
        name = "plot"
    plt.savefig(os.path.join(parent, f"{name}.eps"), bbox_inches="tight")

    # Print results.
    for effect_name in effect_vars:
        for effect_val in total_effect_vals[effect_name]:
            print(effect_val)
            for alg in ALGS:
                msg = f"{NAMES[alg]}"
                for metric in METRICS:
                    metric_val = BEST[metric](total_results[effect_name][effect_val][alg][metric])
                    msg += f" | {NAMES[metric]}: {metric_val:.5f}"
                print(msg)
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family_dirs", nargs='*', help="Folders whose subdirectories are results for one value of effecting variable.",
    )
    parser.add_argument(
        "name", type=str, help="Name to store plot under."
    )
    args = parser.parse_args()
    plot(args.family_dirs, args.name)
