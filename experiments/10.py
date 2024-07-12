"""
Comparison of route anonymity and configuration utility between ConfMask and strawmans.
"""

import json

import click
import matplotlib.pyplot as plt
import rich

import shared
from config import (
    ANONYM_NAME,
    RESULTS_DIR,
    NETWORKS_DIR,
    STATS_FILE,
    ALGORITHMS,
    ALGORITHM_LABELS,
)


@click.command()
@shared.cli_network()
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
def main(network, kr, kh, seed):
    shared.display_title("Figure 10", kr=kr, kh=kh, seed=seed)
    all_results = {}  # algorithm -> (route anonymity, config utility)
    anonymity_missing = []  # list of missing algorithms
    utility_missing = []  # list of missing algorithms

    for algorithm in ALGORITHMS:
        target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
        cur_anonymity, cur_utility = None, None

        # Try to load anonymity
        results_file = RESULTS_DIR / f"5-{target}.json"
        if not results_file.exists():
            anonymity_missing.append(algorithm)
        else:
            with results_file.open("r", encoding="utf-8") as f:
                results = json.load(f)
            cur_anonymity = results.get(network)
            if cur_anonymity is None:
                anonymity_missing.append(algorithm)

        # Try to load utility
        stats_file = NETWORKS_DIR / network / target / STATS_FILE
        if not stats_file.exists():
            utility_missing.append(algorithm)
        else:
            with stats_file.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            cur_utility = 1 - sum(stats["config_lines_modified"].values()) / sum(
                stats["config_lines_total"].values()
            )

        if cur_anonymity is not None and cur_utility is not None:
            all_results[algorithm] = (cur_anonymity, cur_utility)

    if len(anonymity_missing) > 0 or len(utility_missing) > 0:
        rich.print("[red]Some data are missing; try running:")
        for missing_algorithm in utility_missing:
            cmd = shared.get_gen_cmd(network, missing_algorithm, kr, kh, seed)
            rich.print(f"[red]>[/red] {cmd}")
        for missing_algorithm in anonymity_missing:
            cmd = shared.get_5_cmd([network], missing_algorithm, kr, kh, seed)
            rich.print(f"[red]>[/red] {cmd}")
        return

    # Plot the graph
    if len(all_results) > 0:
        save_name = ANONYM_NAME.format(algorithm="all", kr=kr, kh=kh, seed=seed)
        alg_labels = [ALGORITHM_LABELS[algorithm] for algorithm in ALGORITHMS]
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(alg_labels, [all_results[alg][0] for alg in ALGORITHMS])
        ax2.bar(alg_labels, [all_results[alg][1] for alg in ALGORITHMS])
        ax1.set_ylabel("Route anonymity")
        ax2.set_ylabel("Configuration utility")
        ax2.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"10-{save_name}-{network}.png")


if __name__ == "__main__":
    main()
