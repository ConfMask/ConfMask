"""
Running time comparision of ConfMask and strawmans.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np

import shared
from config import (
    RESULTS_DIR,
    STATS_FILE,
    NETWORKS_DIR,
    ALGORITHMS,
    ANONYM_NAME,
    ALGORITHM_LABELS,
)


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
def main(networks, kr, kh, seed):
    shared.display_title("Figure 16", kr=kr, kh=kh, seed=seed)
    all_results = {}  # (algorithm, network) -> time
    missing = defaultdict(list)  # algorithm -> list of missing networks

    for algorithm in ALGORITHMS:
        for network in networks:
            target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
            stats_file = NETWORKS_DIR / network / target / STATS_FILE
            if not stats_file.exists():
                missing[algorithm].append(network)
            else:
                with stats_file.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
                all_results[(algorithm, network)] = stats["time_elapsed"]

    if len(missing) > 0:
        shared.display_cmd_hints(
            [
                ("gen", missing_networks, algorithm, kr, kh, seed)
                for algorithm, missing_networks in missing.items()
            ]
        )
        return

    # Plot the graph
    if len(all_results) > 0:
        save_name = ANONYM_NAME.format(algorithm="all", kr=kr, kh=kh, seed=seed)
        x, width = np.arange(len(networks)), 0.8 / len(ALGORITHMS)
        plt.figure()
        for i, algorithm in enumerate(ALGORITHMS):
            plt.bar(
                x + i * width,
                [all_results[(algorithm, network)] for network in networks],
                width,
                label=ALGORITHM_LABELS[algorithm],
            )
        plt.ylabel("Running time (s)")
        plt.yscale("log")
        plt.xticks(
            x + width * (len(ALGORITHMS) - 1) / 2,
            [f"Net{network}" for network in networks],
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"16-{save_name}.png")


if __name__ == "__main__":
    main()
