"""
Impact of kR on configuration utility.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

from config import CONFMASK_NAME, RESULTS_DIR, STATS_FILE, NETWORKS_DIR

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
@click.option(
    "--kr", required=True, type=int, multiple=True, help="Router anonymization degree."
)
@click.option("--kh", required=True, type=int, help="Host anonymization degree.")
@click.option("--seed", required=True, type=int, help="Random seed.")
def main(networks, kr, kh, seed):
    rich.get_console().rule(f"Figure 13 | kr={','.join(map(str, kr))}, {kh=}, {seed=}")
    all_results = {}  # (kr, kh, network) -> config utility
    missing = defaultdict(list)  # (kr, kh) -> list of missing networks
    names = sorted(set(SUPPORTED_NETWORKS) & set(networks))

    for _kr in kr:
        target = CONFMASK_NAME.format(kr=_kr, kh=kh, seed=seed)
        for name in names:
            stats_path = NETWORKS_DIR / name / target / STATS_FILE
            if not stats_path.exists():
                missing[(_kr, kh)].append(name)
            else:
                with stats_path.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
                all_results[(_kr, kh, name)] = 1 - sum(
                    stats["config_lines_modified"].values()
                ) / sum(stats["config_lines_total"].values())

    if len(missing) > 0:
        rich.print("[red]Some data are missing; try running:")
        for (_kr, kh), names in missing.items():
            for name in names:
                rich.print(
                    f"[red]>[/red] python experiments/gen.py --kr {_kr} --kh {kh} --seed {seed} -n {name}"
                )
        return

    # Plot the graph
    if len(all_results) > 0:
        x, width = np.arange(len(names)), 0.8 / len(kr)
        plt.figure()
        for i, _kr in enumerate(kr):
            plt.bar(
                x + i * width,
                [all_results[(_kr, kh, name)] for name in names],
                width,
                label=f"kR={_kr}",
            )
        plt.ylim(0, 1)
        plt.ylabel("Configuration utility")
        plt.xticks(x + width * (len(kr) - 1) / 2, [f"Net{name}" for name in names])
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR
            / f"13-{CONFMASK_NAME.format(kr='_'.join(map(str, kr)), kh=kh, seed=seed)}.png"
        )


if __name__ == "__main__":
    main()
