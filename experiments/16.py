"""
Running time comparision of ConfMask and strawmans.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

from config import (
    RESULTS_DIR,
    STATS_FILE,
    NETWORKS_DIR,
    PROTOCOL_MAPPING,
    ALGORITHMS,
    ANONYM_NAME,
)


@click.command()
@click.option(
    "-n",
    "--network",
    required=True,
    type=click.Choice(sorted(PROTOCOL_MAPPING)),
    multiple=True,
    help="Networks to evaluate.",
)
@click.option("--kr", required=True, type=int, help="Router anonymization degree.")
@click.option("--kh", required=True, type=int, help="Host anonymization degree.")
@click.option("--seed", required=True, type=int, help="Random seed.")
def main(network, kr, kh, seed):
    rich.get_console().rule(f"Figure 16 | {kr=}, {kh=}, {seed=}")
    all_results = {}  # (algorithm, network) -> time
    missing = defaultdict(list)  # algorithm -> list of missing networks

    for algorithm in ALGORITHMS:
        for _network in network:
            target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
            stats_path = NETWORKS_DIR / _network / target / STATS_FILE
            if not stats_path.exists():
                missing[algorithm].append(_network)
            else:
                with stats_path.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
                all_results[(algorithm, _network)] = stats["time_elapsed"]

    if len(missing) > 0:
        rich.print("[red]Some data are missing; try running:")
        for algorithm, missing_networks in missing.items():
            for missing_network in missing_networks:
                rich.print(
                    f"[red]>[/red] python experiments/gen.py --kr {kr} --kh {kh} --seed {seed} -n {missing_network} -a {algorithm}"
                )
        return

    # Plot the graph
    if len(all_results) > 0:
        x, width = np.arange(len(network)), 0.8 / len(ALGORITHMS)
        plt.figure()
        for i, algorithm in enumerate(ALGORITHMS):
            plt.bar(
                x + i * width,
                [all_results[(algorithm, _network)] for _network in network],
                width,
                label=algorithm.capitalize(),
            )
        plt.ylabel("Running time (s)")
        plt.yscale("log")
        plt.xticks(
            x + width * (len(ALGORITHMS) - 1) / 2,
            [f"Net{_network}" for _network in network],
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR
            / f"16-{ANONYM_NAME.format(algorithm='all', kr=kr, kh=kh, seed=seed)}.png"
        )


if __name__ == "__main__":
    main()
