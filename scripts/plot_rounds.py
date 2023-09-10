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
    "SNLI": "SNLI",
    "Sent140": "Sentiment140",
}
PLOT_SIZE = 4
LINE_WIDTH = 1.5
HORIZONTAL_STRETCH = 1.0


plt.rcParams.update({'font.size': 16})


def plot(family_dirs, name):

    # Read results for all directories.
    results = {}
    rounds = {}
    for family_dir in family_dirs:

        # Read results of each run (averaged over workers).
        results[family_dir] = {}
        rounds[family_dir] = {}
        for alg in ALGS:
            alg_files = glob.glob(os.path.join(family_dir, alg, "*.json"))
            candidate_avg = [filename for filename in alg_files if "Rank" not in filename]
            if len(candidate_avg) != 1:
                print(f"Incomplete results for {alg}, skipping.")
                continue
            avg_filename = candidate_avg[0]
            with open(os.path.join(avg_filename), "r") as f:
                results[family_dir][alg] = json.load(f)

            # Get interval and rounds.
            rounds_key = "_Rounds_"
            rounds_start = avg_filename.find(rounds_key) + len(rounds_key)
            rounds_end = avg_filename.find("_", rounds_start)
            rounds[family_dir][alg] = int(avg_filename[rounds_start: rounds_end])

    # Make subplots.
    nplots = len(family_dirs) * len(METRICS)
    fig, axs = plt.subplots(nrows=1, ncols=nplots)
    fig.set_size_inches(
        nplots * PLOT_SIZE * HORIZONTAL_STRETCH, PLOT_SIZE
    )
    i = 0
    for family_dir in family_dirs:

        start = family_dir.find("_") + 1
        end = family_dir.find("_", start)
        dataset = family_dir[start:end]
        min_rounds = np.min(list(rounds[family_dir].values()))
        for k, metric in enumerate(METRICS):

            y_min = float("inf")
            y_max = float("-inf")

            x = np.arange(min_rounds)
            for alg in ALGS:
                current_rounds = rounds[family_dir][alg]
                num_evals = len(results[family_dir][alg][metric])
                eval_rounds = [
                    round((j+1) * (current_rounds - 1) / num_evals)
                    for j in range(num_evals)
                ]
                x = np.array([r for r in eval_rounds if r <= min_rounds])
                y = results[family_dir][alg][metric][:len(x)]
                plot_kwargs = {"linewidth": LINE_WIDTH}
                if i == 0:
                    plot_kwargs["label"] = NAMES[alg]
                axs[i].plot(x, y, **plot_kwargs)

                y_min = min(y_min, np.min(y))
                y_max = max(y_max, np.max(y))

            y_min -= (y_max - y_min) * 0.05
            y_max += (y_max - y_min) * 0.05
            if k == 0:
                axs[i].set_title(NAMES[dataset])
            axs[i].set_xlabel("Rounds")
            axs[i].set_ylabel(NAMES[metric])
            axs[i].set_ylim([y_min, y_max])

            i += 1

    #fig.suptitle(name, y=0.9)
    plt.figlegend(loc="lower center", ncol=len(ALGS), bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    parent = os.path.normpath(os.path.join(family_dirs[0], "../.."))
    plt.savefig(os.path.join(parent, f"{name}.eps"), bbox_inches="tight")

    # Print results.
    print(os.path.basename(family_dir))
    print(f"Best metric values during training:")
    for family_dir in family_dirs:
        print(family_dir)
        for alg in ALGS:
            msg = f"{alg}"
            for metric in METRICS:
                metric_val = BEST[metric](results[family_dir][alg][metric])
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
