"""
Impact of kR on route anonymity.
"""

import json
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

from config import CONFMASK_NAME, RESULTS_DIR

SUPPORTED_NETWORKS = "ADEH"


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
    rich.get_console().rule(f"Figure 11 | kr={','.join(map(str, kr))}, {kh=}, {seed=}")
    all_results = {}  # (kr, kh, network) -> route anonymity
    missing = defaultdict(list)  # (kr, kh) -> list of missing networks
    names = sorted(set(SUPPORTED_NETWORKS) & set(networks))

    for _kr in kr:
        target = CONFMASK_NAME.format(kr=_kr, kh=kh, seed=seed)
        results_file = RESULTS_DIR / f"5-{target}.json"
        if not results_file.exists():
            missing[(_kr, kh)] = names
        else:
            with results_file.open("r", encoding="utf-8") as f:
                results = json.load(f)
            for name in names:
                if name not in results:
                    missing[(_kr, kh)].append(name)
                else:
                    all_results[(_kr, kh, name)] = results[name]

    if len(missing) > 0:
        rich.print("[red]Data are missing from Experiment 5; try running:")
        for (_kr, kh), names in missing.items():
            rich.print(
                f"[red]>[/red] python experiments/5.py --kr {_kr} --kh {kh} --seed {seed} -n {''.join(names)}"
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
        plt.ylabel("Route anonymity")
        plt.xticks(x + width * (len(kr) - 1) / 2, [f"Net{name}" for name in names])
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR
            / f"11-{CONFMASK_NAME.format(kr='_'.join(map(str, kr)), kh=kh, seed=seed)}.png"
        )


if __name__ == "__main__":
    main()
