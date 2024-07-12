"""
Impact of kH on route anonymity.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np

import shared
from config import ANONYM_NAME, RESULTS_DIR


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh(multiple=True)
@shared.cli_seed()
def main(networks, algorithm, kr, khs, seed):
    shared.display_title(
        "Figure 12", algorithm=algorithm, kr=kr, kh=",".join(map(str, khs)), seed=seed
    )
    all_results = {}  # (kr, kh, network) -> route anonymity
    missing = defaultdict(list)  # (kr, kh) -> list of missing networks
    networks = sorted(networks)

    for kh in khs:
        target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
        results_file = RESULTS_DIR / f"5-{target}.json"
        if not results_file.exists():
            missing[(kr, kh)] = networks
        else:
            with results_file.open("r", encoding="utf-8") as f:
                results = json.load(f)
            for network in networks:
                if network not in results:
                    missing[(kr, kh)].append(network)
                else:
                    all_results[(kr, kh, network)] = results[network]

    if len(missing) > 0:
        shared.display_cmd_hints(
            [
                ("5", missing_networks, algorithm, kr, kh, seed)
                for (kr, kh), missing_networks in missing.items()
            ]
        )
        return

    # Plot the graph
    if len(all_results) > 0:
        save_name = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh="", seed=seed)
        x, width = np.arange(len(networks)), 0.8 / len(khs)
        plt.figure()
        for i, kh in enumerate(khs):
            plt.bar(
                x + i * width,
                [all_results[(kr, kh, network)] for network in networks],
                width,
                label=f"kH={kh}",
            )
        plt.ylabel("Route anonymity")
        plt.xticks(
            x + width * (len(khs) - 1) / 2, [f"Net{network}" for network in networks]
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"12-{save_name}.png")


if __name__ == "__main__":
    main()
