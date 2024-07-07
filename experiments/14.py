"""
Impact of kH on configuration utility.
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
    "-n",
    "--networks",
    type=str,
    default=SUPPORTED_NETWORKS,
    show_default=True,
    help="Networks to evaluate.",
)
@click.option("--kr", required=True, type=int, help="Router anonymization degree.")
@click.option(
    "--kh", required=True, type=int, multiple=True, help="Host anonymization degree."
)
@click.option("--seed", required=True, type=int, help="Random seed.")
def main(networks, kr, kh, seed):
    rich.get_console().rule(f"Figure 14 | {kr=}, kh={','.join(map(str, kh))}, {seed=}")
    all_results = {}  # (kr, kh, network) -> config utility
    missing = defaultdict(list)  # (kr, kh) -> list of missing networks
    names = sorted(set(SUPPORTED_NETWORKS) & set(networks))

    for _kh in kh:
        target = CONFMASK_NAME.format(kr=kr, kh=_kh, seed=seed)
        for name in names:
            stats_path = NETWORKS_DIR / name / target / STATS_FILE
            if not stats_path.exists():
                missing[(kr, _kh)].append(name)
            else:
                with stats_path.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
                all_results[(kr, _kh, name)] = 1 - sum(
                    stats["config_lines_modified"].values()
                ) / sum(stats["config_lines_total"].values())

    if len(missing) > 0:
        rich.print("[red]Some data are missing; try running:")
        for (kr, _kh), names in missing.items():
            for name in names:
                rich.print(
                    f"[red]>[/red] python experiments/gen.py --kr {kr} --kh {_kh} --seed {seed} -n {name}"
                )
        return

    # Plot the graph
    if len(all_results) > 0:
        x, width = np.arange(len(names)), 0.8 / len(kh)
        plt.figure()
        for i, _kh in enumerate(kh):
            plt.bar(
                x + i * width,
                [all_results[(kr, _kh, name)] for name in names],
                width,
                label=f"kH={_kh}",
            )
        plt.ylabel("Configuration utility")
        plt.xticks(x + width * (len(kh) - 1) / 2, [f"Net{name}" for name in names])
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR
            / f"14-{CONFMASK_NAME.format(kr=kr, kh='_'.join(map(str, kh)), seed=seed)}.png"
        )


if __name__ == "__main__":
    main()
