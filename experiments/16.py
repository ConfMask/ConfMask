"""
Running time comparision of ConfMask and strawmans.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

import shared
from config import RESULTS_DIR, STATS_FILE, NETWORKS_DIR, ALGORITHMS, ANONYM_NAME


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
def main(networks, kr, kh, seed):
    rich.get_console().rule(f"Figure 16 | {kr=}, {kh=}, {seed=}")
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
        rich.print("[red]Some data are missing; try running:")
        for algorithm, missing_networks in missing.items():
            for missing_network in missing_networks:
                cmd = shared.get_gen_cmd(missing_network, algorithm, kr, kh, seed)
                rich.print(f"[red]>[/red] {cmd}")
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
                label=algorithm.capitalize(),
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
