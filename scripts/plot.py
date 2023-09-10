""" Plot training results. """

import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


LINE_WIDTH = 1.5
ROUNDS_PER_EVAL = 138
IGNORE_METRICS = ["eval_elasped_times", "clip_ops"]
YLIM_WINDOW_START = 0.25
AVG_WINDOW_START = 0.9
#DISPLAY_NAMES = {"local_clip_0.4": "CELGC", "naive_parallel_0.4": "Naive Parallel Clip", "episode_0.4": "EPISODE", "train_losses": "Train Loss", "test_accuracies": "Test Accuracy", "test_losses": "Test Loss", "train_accuracies": "Train Accuracy", "local_train_losses": "Local Train Loss", "local_train_accuracies": "Local Train Accuracy"}
plt.rcParams.update({'font.size': 14, 'font.family': "FreeSerif"})


def plot(family_dir):

    # Read results of each run (averaged over workers).
    results = {}
    metrics = None
    for run_path, _, run_files in list(os.walk(family_dir))[1:]:
        run_name = os.path.basename(run_path)
        candidate_avg = [filename for filename in run_files if "Rank" not in filename and filename.endswith(".json")]
        if len(candidate_avg) != 1:
            print(f"Incomplete results for {run_name}, skipping.")
            continue
        avg_filename = candidate_avg[0]
        with open(os.path.join(run_path, avg_filename), "r") as f:
            results[run_name] = json.load(f)

        if metrics is None:
            metrics = list(results[run_name].keys())
        else:
            assert metrics == list(results[run_name].keys())

    if metrics is None:
        return

    # Re-order methods.
    names = list(results.keys())

    # Plot results.
    for metric in metrics:
        if metric in IGNORE_METRICS:
            continue
        plt.clf()
        y_min = None
        y_max = None
        for name in names:
            x = np.arange(len(results[name][metric])) * ROUNDS_PER_EVAL
            plt.plot(x, results[name][metric], label=name, linewidth=LINE_WIDTH)
            start = round(len(results[name][metric]) * YLIM_WINDOW_START)
            current_min = float(np.min(results[name][metric][start:]))
            current_max = float(np.max(results[name][metric][start:]))
            if y_min is None:
                y_min = current_min
                y_max = current_max
            else:
                y_min = min(y_min, current_min)
                y_max = max(y_max, current_max)

        plt.xlabel("Rounds")
        plt.ylabel(metric)
        plt.legend()
        y_min -= (y_max - y_min) * 0.05
        y_max += (y_max - y_min) * 0.05
        plt.ylim([y_min, y_max])

        plt.savefig(os.path.join(family_dir, f"{metric}.png"), bbox_inches="tight")

    # Print results.
    window_percent = round((1 - AVG_WINDOW_START) * 100)
    print(os.path.basename(family_dir))
    print(f"Average metric values over last {window_percent}% of training:")
    for name in results:
        msg = f"{name}"
        for metric in metrics:
            if metric not in IGNORE_METRICS:
                start = round(len(results[name][metric]) * AVG_WINDOW_START)
                window_mean = float(np.mean(results[name][metric][start:]))
                msg += f" | {metric}: {window_mean:.5f}"
        print(msg)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family_dirs", nargs="*", help="Folders whose subdirectories are results for individual runs to compare",
    )
    args = parser.parse_args()
    for family_dir in args.family_dirs:
        plot(family_dir)
