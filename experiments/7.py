"""
Clustering coefficients of all networks.
"""

import json

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from confmask.utils import analyze_topology
from pybatfish.client.session import Session

import shared
from config import (
    NETWORKS_DIR,
    ANONYM_NAME,
    RESULTS_DIR,
    ORIGIN_NAME,
    BF_HOST,
    ALGORITHM_LABELS,
)

bf = Session(host=BF_HOST)


def run_network(network, algorithm, target, progress, task):
    """Execute the experiment for a single network."""

    def _display(**kwargs):
        progress.update(task, **kwargs)

    def _get_least_anonymized(ver):
        """Get the least anonymized node degree."""
        _display(description="Uploading configurations...")
        bf.set_network(network)
        bf.init_snapshot(str(NETWORKS_DIR / network / ver), name=ver, overwrite=True)
        _display(description="Querying topology...")
        topology = bf.q.layer3Edges().answer().frame()
        _display(description="Processing...")
        _, _, _, _, _, nx_graph = analyze_topology(topology)
        return nx.average_clustering(nx_graph)

    _display(network=f"[{network}/Original]")
    coef_origin = _get_least_anonymized(ORIGIN_NAME)
    _display(network=f"[{network}/{ALGORITHM_LABELS[algorithm]}]")
    coef_target = _get_least_anonymized(target)

    _display(
        network=f"[{network}]",
        description=f"[bold green]Done[/bold green] | {coef_origin:.3f} -> {coef_target:.3f}",
    )
    return coef_origin, coef_target


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
@shared.cli_plot_only()
def main(networks, algorithm, kr, kh, seed, plot_only):
    shared.display_title("Figure 07", algorithm=algorithm, kr=kr, kh=kh, seed=seed)
    results = {}
    target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
    networks = sorted(networks) if not plot_only else []

    def _run_network_func(network, *, progress, task):
        results[network] = run_network(network, algorithm, target, progress, task)

    shared.display_progress(networks, _run_network_func)

    # Merge results with existing (if any)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"7-{target}.json"
    all_results = {}
    if results_file.exists():
        with results_file.open("r", encoding="utf-8") as f:
            all_results = json.load(f)
    all_results.update(results)
    if not plot_only:
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # Plot the graph
    if len(all_results) > 0:
        all_results = sorted(all_results.items())
        x, width = np.arange(len(all_results)), 0.4
        plt.figure()
        plt.bar(x, [v for _, (v, _) in all_results], width, label="Original")
        plt.bar(
            x + width,
            [v for _, (_, v) in all_results],
            width,
            label=ALGORITHM_LABELS[algorithm],
        )
        plt.ylabel("Clustering coefficient")
        plt.ylim(0, 1)
        plt.xticks(x + width / 2, [f"Net{k}" for k, _ in all_results])
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"7-{target}.png")


if __name__ == "__main__":
    main()
