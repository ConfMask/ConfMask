"""
Tradeoff between route anonymity and configuration utility.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import shared
from config import ANONYM_NAME, NETWORKS_DIR, RESULTS_DIR, STATS_FILE


@click.command()
@click.option(
    "-c",
    "--cases",
    required=True,
    multiple=True,
    help="Cases to run, each in the form 'kr,kh,network'.",
)
@shared.cli_algorithm()
@shared.cli_seed()
def main(cases, algorithm, seed):
    shared.display_title("Figure 15", algorithm=algorithm, seed=seed)
    all_results = {}  # (kr, kh, network) -> (route anonymity, config utility)
    missing_anonymity = defaultdict(list)  # (kr, kh) -> list of missing networks
    missing_utility = defaultdict(list)  # (kr, kh) -> list of missing networks

    for case in cases:
        kr, kh, network = case.split(",")
        kr, kh = int(kr), int(kh)
        target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
        cur_anonymity, cur_utility = None, None

        # Try to load the route anonymity
        results_file = RESULTS_DIR / f"5-{target}.json"
        if not results_file.exists():
            missing_anonymity[(kr, kh)].append(network)
        else:
            with results_file.open("r", encoding="utf-8") as f:
                results = json.load(f)
            anonymity = results.get(network)
            if anonymity is None:
                missing_anonymity[(kr, kh)].append(network)
            else:
                cur_anonymity = anonymity

        # Try to load the configuration utility
        stats_file = NETWORKS_DIR / network / target / STATS_FILE
        if not stats_file.exists():
            missing_utility[(kr, kh)].append(network)
        else:
            with stats_file.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            cur_utility = 1 - sum(stats["config_lines_modified"].values()) / sum(
                stats["config_lines_total"].values()
            )

        if cur_anonymity is not None and cur_utility is not None:
            all_results[(kr, kh, network)] = (cur_anonymity, cur_utility)

    if len(missing_anonymity) > 0 or len(missing_utility) > 0:
        shared.display_cmd_hints(
            [
                ("gen", missing_networks, algorithm, kr, kh, seed)
                for (kr, kh), missing_networks in missing_utility.items()
            ]
            + [
                ("5", missing_networks, algorithm, kr, kh, seed)
                for (kr, kh), missing_networks in missing_anonymity.items()
            ]
        )
        return

    # Plot the graph and save the statistics
    if len(all_results) > 0:
        save_name = ANONYM_NAME.format(algorithm=algorithm, kr="", kh="", seed=seed)
        all_results = sorted(all_results.items(), key=lambda x: x[0])
        anonymities = [item[1][0] for item in all_results]
        utilities = [item[1][1] for item in all_results]
        plt.figure()
        plt.scatter(anonymities, utilities, s=80, alpha=0.4)
        plt.ylim(0, 1)
        plt.xlabel("Route anonymity")
        plt.ylabel("Configuration utility")
        plt.grid(alpha=0.6)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"15-{save_name}.png")

        pearson_correlation = np.corrcoef(anonymities, utilities)[0, 1]
        with (RESULTS_DIR / f"15-{save_name}.json").open("w", encoding="utf-8") as f:
            json.dump({"pearson_correlation": pearson_correlation}, f, indent=2)


if __name__ == "__main__":
    main()
