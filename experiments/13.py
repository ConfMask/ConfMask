"""
Impact of kR on configuration utility.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

import shared
from config import ANONYM_NAME, RESULTS_DIR, STATS_FILE, NETWORKS_DIR


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr(multiple=True)
@shared.cli_kh()
@shared.cli_seed()
def main(networks, algorithm, krs, kh, seed):
    shared.display_title(
        13, algorithm=algorithm, kr=",".join(map(str, krs)), kh=kh, seed=seed
    )
    all_results = {}  # (kr, kh, network) -> config utility
    missing = defaultdict(list)  # (kr, kh) -> list of missing networks
    networks = sorted(networks)

    for kr in krs:
        target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
        for network in networks:
            stats_file = NETWORKS_DIR / network / target / STATS_FILE
            if not stats_file.exists():
                missing[(kr, kh)].append(network)
            else:
                with stats_file.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
                all_results[(kr, kh, network)] = 1 - sum(
                    stats["config_lines_modified"].values()
                ) / sum(stats["config_lines_total"].values())

    if len(missing) > 0:
        rich.print("[red]Some data are missing; try running:")
        for (kr, kh), missing_networks in missing.items():
            for missing_network in missing_networks:
                cmd = shared.get_gen_cmd(missing_network, algorithm, kr, kh, seed)
                rich.print(f"[red]>[/red] {cmd}")
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
        plt.ylim(0, 1)
        plt.ylabel("Configuration utility")
        plt.xticks(
            x + width * (len(krs) - 1) / 2, [f"Net{network}" for network in networks]
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"13-{save_name}.png")


if __name__ == "__main__":
    main()
