"""
Clustering coefficients of all networks.
"""

import json

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rich
from confmask.utils import analyze_topology
from rich.progress import Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session

import shared
from config import NETWORKS_DIR, ANONYM_NAME, RESULTS_DIR, ORIGIN_NAME, BF_HOST

bf = Session(host=BF_HOST)


def run_network(network, algorithm, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)

    def _phase(description):
        progress.update(task, description=f"[{network}] {description}")

    def _get_least_anonymized(ver, prefix):
        """Get the least anonymized node degree."""
        _phase(f"[{prefix}] Uploading configurations...")
        bf.set_network(network)
        bf.init_snapshot(str(NETWORKS_DIR / network / ver), name=ver, overwrite=True)
        _phase(f"[{prefix}] Querying topology...")
        topology = bf.q.layer3Edges().answer().frame()
        _phase(f"[{prefix}] Processing...")
        _, _, _, _, _, nx_graph = analyze_topology(topology)
        return nx.average_clustering(nx_graph)

    coef_origin = _get_least_anonymized(ORIGIN_NAME, "Original")
    coef_target = _get_least_anonymized(target, algorithm.capitalize())

    _phase(f"[green]Done[/green] | {coef_origin:.3f} -> {coef_target:.3f}")
    progress.stop_task(task)
    return coef_origin, coef_target


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
@shared.cli_plot_only()
def main(networks, algorithm, kr, kh, seed, plot_only):
    rich.get_console().rule(f"Figure 7 | {kr=}, {kh=}, {seed=}")
    results = {}
    target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
    networks = sorted(networks) if not plot_only else []

    if len(networks) > 0:
        with Progress(
            TimeElapsedColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
        ) as progress:
            tasks = {
                network: progress.add_task(
                    f"[{network}] (queued)", start=False, total=None
                )
                for network in networks
            }
            for network in networks:
                result = run_network(
                    network, algorithm, target, progress, tasks[network]
                )
                results[network] = result

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
            label=algorithm.capitalize(),
        )
        plt.ylabel("Clustering coefficient")
        plt.ylim(0, 1)
        plt.xticks(x + width / 2, [f"Net{k}" for k, _ in all_results])
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"7-{target}.png")


if __name__ == "__main__":
    main()
