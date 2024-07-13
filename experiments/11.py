"""
Impact of kR on route anonymity.
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
@shared.cli_kr(multiple=True)
@shared.cli_kh()
@shared.cli_seed()
def main(networks, algorithm, krs, kh, seed):
    shared.display_title(
        "Figure 11", algorithm=algorithm, kr=",".join(map(str, krs)), kh=kh, seed=seed
    )
    all_results = {}  # (kr, kh, network) -> route anonymity
    missing = defaultdict(list)  # (kr, kh) -> list of missing networks
    networks = sorted(networks)

    for kr in krs:
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
        save_name = ANONYM_NAME.format(algorithm=algorithm, kr="", kh=kh, seed=seed)
        x, width = np.arange(len(networks)), 0.8 / len(krs)
        plt.figure()
        for i, kr in enumerate(krs):
            plt.bar(
                x + i * width,
                [all_results[(kr, kh, network)] for network in networks],
                width,
                label=f"kR={kr}",
            )
        plt.ylabel("Route anonymity")
        plt.xticks(
            x + width * (len(krs) - 1) / 2, [f"Net{network}" for network in networks]
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"11-{save_name}.png")


if __name__ == "__main__":
    main()
