"""
Comparison of route anonymity and configuration utility between ConfMask and strawmans.
"""

import json

import click
import matplotlib.pyplot as plt
import rich

from config import (
    ANONYM_NAME,
    RESULTS_DIR,
    PROTOCOL_MAPPING,
    NETWORKS_DIR,
    STATS_FILE,
    ALGORITHMS,
)


@click.command()
@click.option(
    "-n",
    "--network",
    required=True,
    type=click.Choice(list(PROTOCOL_MAPPING)),
    help="Network to evaluate.",
)
@click.option("--kr", required=True, type=int, help="Router anonymization degree.")
@click.option("--kh", required=True, type=int, help="Host anonymization degree.")
@click.option("--seed", required=True, type=int, help="Random seed.")
def main(network, kr, kh, seed):
    rich.get_console().rule(f"Figure 10 | {kr=}, {kh=}, {seed=}")
    all_results = {}  # algorithm -> (route anonymity, config utility)
    anonymity_missing = []  # list of algorithms
    utility_missing = []  # list of algorithms

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
        stats_path = NETWORKS_DIR / network / target / STATS_FILE
        if not stats_path.exists():
            utility_missing.append(algorithm)
        else:
            with stats_path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            cur_utility = 1 - sum(stats["config_lines_modified"].values()) / sum(
                stats["config_lines_total"].values()
            )

        if cur_anonymity is not None and cur_utility is not None:
            all_results[algorithm] = (cur_anonymity, cur_utility)

    if len(anonymity_missing) > 0 or len(utility_missing) > 0:
        rich.print("[red]Some data are missing; try running:")
        for algorithm in utility_missing:
            rich.print(
                f"[red]>[/red] python experiments/gen.py --kr {kr} --kh {kh} --seed {seed} -n {network} -a {algorithm}"
            )
        for algorithm in anonymity_missing:
            rich.print(
                f"[red]>[/red] python experiments/5.py --kr {kr} --kh {kh} --seed {seed} -n {network} -a {algorithm}"
            )
        return

    # Plot the graph
    if len(all_results) > 0:
        alg_labels = [algorithm.capitalize() for algorithm in ALGORITHMS]
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.bar(alg_labels, [all_results[alg][0] for alg in ALGORITHMS])
        ax2.bar(alg_labels, [all_results[alg][1] for alg in ALGORITHMS])
        ax1.set_title("Route anonymity")
        ax2.set_title("Configuration utility")
        ax2.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR
            / f"10-{ANONYM_NAME.format(algorithm='all', kr=kr, kh=kh, seed=seed)}-{network}.png"
        )


if __name__ == "__main__":
    main()
