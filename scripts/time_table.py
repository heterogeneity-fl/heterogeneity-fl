""" Create a table of running times for each algorithm. """

import os
import argparse
import json
import glob

import numpy as np


ALGS = ["episode_mem", "local_clip", "naive_parallel_clip", "minibatch_clip", "scaffold_clip", "episode"]
EXTREME = 0.1


def get_time_temp(log_dirs):
    """ Read logs and output time taken by each algorithm. """

    # Read times for each setting and algorithm.
    total_times = {alg: 0 for alg in ALGS}
    for log_dir in log_dirs:
        for alg in ALGS:

            # Read times from log.
            candidate_avg = glob.glob(os.path.join(log_dir, alg, "*_Test.json"))
            assert len(candidate_avg) == 1
            avg_path = candidate_avg[0]
            with open(avg_path, "r") as f:
                alg_results = json.load(f)

            # Throw away extremes and take average.
            times = np.array(alg_results["eval_elasped_times"])
            times = np.sort(times)
            start = round(len(times) * EXTREME)
            end = round(len(times) * (1 - EXTREME))
            times = times[start:end]
            total_times[alg] += float(np.mean(times))

    # Compute and print normalized times.
    smallest_time = min(total_times.values())
    for alg in ALGS:
        print(f"{alg}: {total_times[alg]:.5f} {total_times[alg] / smallest_time:.5f}")


def get_time(log_dirs):
    """ Read logs and output time taken by each algorithm. """

    # Read times for each setting and algorithm.
    final_times = {alg: [] for alg in ALGS}
    for log_dir in log_dirs:
        alg_times = {}
        for alg in ALGS:

            # Read times from log.
            candidate_avg = glob.glob(os.path.join(log_dir, alg, "*_Test.json"))
            assert len(candidate_avg) == 1
            avg_path = candidate_avg[0]
            with open(avg_path, "r") as f:
                alg_results = json.load(f)

            # Throw away extremes and take average.
            times = np.array(alg_results["eval_elasped_times"])
            times = np.sort(times)
            start = round(len(times) * EXTREME)
            end = round(len(times) * (1 - EXTREME))
            times = times[start:end]
            alg_times[alg] = np.mean(times)

        # Find minimum for current run.
        print("===============")
        print(log_dir)
        print(alg_times)
        sorted_times = sorted(alg_times.items(), key=(lambda x: x[1]))
        print(sorted_times[0])

        # Computed normalized times for each algorithm.
        min_time = sorted_times[0][1]
        for alg in ALGS:
            final_times[alg].append(alg_times[alg] / min_time)

    # Average normalized times for each algorithm across all runs.
    for alg in ALGS:
        final_times[alg] = np.mean(final_times[alg])
    print("================")
    print(final_times)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dirs", nargs="*", help="Folders whose subdirectories are results for a single training run, all algorithms.")
    args = parser.parse_args()

    get_time_temp(args.log_dirs)
