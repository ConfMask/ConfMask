"""
Tradeoff between route anonymity and configuration utility.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

from config import CONFMASK_NAME, RESULTS_DIR, NETWORKS_DIR, STATS_FILE

SUPPORTED_NETWORKS = "ADEG"


@click.command()
@click.option(
    "-c",
    "--case",
    required=True,
    multiple=True,
    help="Case in the format kr,kh,network.",
)
@click.option("--seed", required=True, type=int, help="Random seed.")
def main(case, seed):
    rich.get_console().rule(f"Figure 15 | {seed=}")
    all_results = {}  # (kr, kh, network) -> (route anonymity, config utility)
    missing_anonymity = defaultdict(list)  # (kr, kh) -> list of missing networks
    missing_utility = defaultdict(list)  # (kr, kh) -> list of missing networks

    for each_case in case:
        kr, kh, name = each_case.split(",")
        kr, kh = int(kr), int(kh)
        target = CONFMASK_NAME.format(kr=kr, kh=kh, seed=seed)
        cur_anonymity, cur_utility = None, None

        # Try to load the route anonymity
        results_file = RESULTS_DIR / f"5-{target}.json"
        if not results_file.exists():
            missing_anonymity[(kr, kh)].append(name)
        else:
            with results_file.open("r", encoding="utf-8") as f:
                results = json.load(f)
            anonymity = results.get(name)
            if anonymity is None:
                missing_anonymity[(kr, kh)].append(name)
            else:
                cur_anonymity = anonymity

        # Try to load the configuration utility
        stats_path = NETWORKS_DIR / name / target / STATS_FILE
        if not stats_path.exists():
            missing_utility[(kr, kh)].append(name)
        else:
            with stats_path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            cur_utility = 1 - sum(stats["config_lines_modified"].values()) / sum(
                stats["config_lines_total"].values()
            )

        if cur_anonymity is not None and cur_utility is not None:
            all_results[(kr, kh, name)] = (cur_anonymity, cur_utility)

    if len(missing_anonymity) > 0 or len(missing_utility) > 0:
        rich.print("[red]Some data are missing; try running:")
        for (kr, kh), names in missing_utility.items():
            for name in names:
                rich.print(
                    f"[red]>[/red] python experiments/gen.py --kr {kr} --kh {kh} --seed {seed} -n {name}"
                )
        for (kr, kh), names in missing_anonymity.items():
            rich.print(
                f"[red]>[/red] python experiments/5.py --kr {kr} --kh {kh} --seed {seed} -n {''.join(names)}"
            )
        return

    # Plot the graph
    if len(all_results) > 0:
        save_name = f"15-{CONFMASK_NAME.format(kr='', kh='', seed=seed)}"
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
        plt.savefig(RESULTS_DIR / f"{save_name}.png")
        with (RESULTS_DIR / f"{save_name}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"pearson_correlation": np.corrcoef(anonymities, utilities)[0, 1]},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
