"""
Impact of kH on configuration utility.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

import shared
from config import ANONYM_NAME, RESULTS_DIR, NETWORKS_DIR, STATS_FILE


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh(multiple=True)
@shared.cli_seed()
def main(networks, algorithm, kr, khs, seed):
    shared.display_title(
        "Figure 14", algorithm=algorithm, kr=kr, kh=",".join(map(str, khs)), seed=seed
    )
    all_results = {}  # (kr, kh, network) -> config utility
    missing = defaultdict(list)  # (kr, kh) -> list of missing networks
    networks = sorted(networks)

    for kh in khs:
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
        plt.ylim(0, 1)
        plt.ylabel("Configuration utility")
        plt.xticks(
            x + width * (len(khs) - 1) / 2, [f"Net{network}" for network in networks]
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"14-{save_name}.png")


if __name__ == "__main__":
    main()
