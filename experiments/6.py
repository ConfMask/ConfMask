"""
Minimum number of nodes of the same degree in all networks.
"""

import json
from collections import Counter

import click
import matplotlib.pyplot as plt
import numpy as np
from confmask.utils import analyze_topology
from rich.progress import Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session

from config import NETWORKS_DIR, CONFMASK_NAME, RESULTS_DIR, ORIGIN_NAME

bf = Session(host="localhost")


def run_network(name, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)

    def _phase(description):
        progress.update(task, description=f"[{name}] {description}")

    def _get_least_anonymized(ver, prefix):
        """Get the least anonymized node degree."""
        _phase(f"[{prefix}] Uploading configurations")
        bf.set_network(name)
        bf.init_snapshot(str(NETWORKS_DIR / name / ver), name=ver, overwrite=True)
        _phase(f"[{prefix}] Querying topology...")
        topology = bf.q.layer3Edges().answer().frame()
        _phase(f"[{prefix}] Processing...")
        _, _, _, _, _, nx_graph = analyze_topology(topology)
        return Counter(d for _, d in nx_graph.degree()).most_common()[-1]

    _, cnt_origin = _get_least_anonymized(ORIGIN_NAME, "Original")
    _, cnt_target = _get_least_anonymized(target, "Anonymized")

    _phase(f"[green]Done[/green] | {cnt_origin} -> {cnt_target}")
    progress.stop_task(task)
    return cnt_origin, cnt_target

@click.command()
@click.option(
    "-n",
    "--networks",
    type=str,
    default="ABCDEFGH",
    show_default=True,
    help="Networks to evaluate.",
)
@click.option("--kr", required=True, type=int, help="Router anonymization degree.")
@click.option("--kh", required=True, type=int, help="Host anonymization degree.")
@click.option("--seed", required=True, type=int, help="Random seed.")
@click.option(
    "--plot-only",
    is_flag=True,
    help="Plot based on stored results without running any evaluation. Ignores -n/--networks.",
)
def main(networks, kr, kh, seed, plot_only):
    results = {}
    target = CONFMASK_NAME.format(kr=kr, kh=kh, seed=seed)
    names = sorted(set("ABCDEFGH") & set(networks)) if not plot_only else []

    if len(names) > 0:
        with Progress(
            TimeElapsedColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
        ) as progress:
            tasks = {
                name: progress.add_task(f"[{name}] (queued)", start=False, total=None)
                for name in names
            }
            for name in names:
                result = run_network(name, target, progress, tasks[name])
                results[name] = result

    # Merge results with existing (if any)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"6-{target}.json"
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
        plt.bar(x + width, [v for _, (_, v) in all_results], width, label="Anonymized")
        plt.ylabel("Min # of same-degree nodes")
        plt.xticks(x + width / 2, [f"Net{k}" for k, _ in all_results])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"6-{target}.png")
        plt.show()


if __name__ == "__main__":
    main()
