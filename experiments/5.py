"""
Average number of distinct paths between edge routers in all networks.
"""

import json
from collections import defaultdict
from itertools import permutations

import click
import matplotlib.pyplot as plt
import rich
from joblib import Parallel, delayed
from rich.progress import Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session
from pybatfish.datamodel.flow import HeaderConstraints

from config import NETWORKS_DIR, RESULTS_DIR, BF_HOST, ANONYM_NAME

SUPPORTED_NETWORKS = "ABCDEFGH"
bf = Session(host=BF_HOST)


def run_network(name, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)

    def _phase(description):
        progress.update(task, description=f"[{name}] {description}")

    _phase("Uploading configurations...")
    bf.set_network(name)
    bf.init_snapshot(str(NETWORKS_DIR / name / target), name=target, overwrite=True)
    _phase("Querying topology...")
    topology = bf.q.layer3Edges().answer().frame()
    _phase("Processing...")

    # Extract information from the network topology
    gw_nodes = defaultdict(list)  # Destination gateway -> source hosts
    gw_interfaces = {}  # Source host -> destination gateway interface
    host_ips = {}  # Host -> IP
    for row in topology.itertuples(index=False):
        src_node, dst_node = (
            row.Interface.hostname,
            row.Remote_Interface.hostname,
        )
        if "host" in src_node:
            gw_nodes[dst_node].append(src_node)
            gw_interfaces[src_node] = row.Remote_Interface.interface
            host_ips[src_node] = row.IPs[0]
        if "host" in dst_node:
            host_ips[dst_node] = row.Remote_IPs[0]

    def _trace(src_gw, dst_gw):
        """Trace routes between two gateways."""
        paths_mem = set()

        # For all possible IPs attached to the source gateway, run traceroute to
        # all possible IPs attached to the destination gateway
        for src_host in gw_nodes[src_gw]:
            for dst_host in gw_nodes[dst_gw]:
                if src_host == dst_host:
                    continue

                start_location = f"@enter({src_gw}[{gw_interfaces[src_host]}])"
                trace_route = (
                    bf.q.traceroute(
                        startLocation=start_location,
                        headers=HeaderConstraints(
                            srcIps=host_ips[src_host],
                            dstIps=host_ips[dst_host],
                        ),
                    )
                    .answer()
                    .frame()
                )
                _phase(f"{host_ips[src_host]} -> {host_ips[dst_host]}")
                path = trace_route.Traces[0][0]
                paths_mem.add("->".join(hop.node for hop in path.hops[:-1]))

        return len(paths_mem)

    gw_pairs, all_num_paths = list(permutations(gw_nodes, 2)), []
    progress.update(task, total=len(gw_pairs))
    for num in Parallel(n_jobs=-1, prefer="threads", return_as="generator_unordered")(
        delayed(_trace)(src_gw, dst_gw) for src_gw, dst_gw in gw_pairs
    ):
        all_num_paths.append(num)
        progress.update(task, advance=1)

    result = sum(all_num_paths) / len(all_num_paths)
    _phase(f"[green]Done[/green] | {result=:.3f}")
    progress.stop_task(task)
    return result


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
    "-a",
    "--algorithm",
    type=click.Choice(["confmask", "strawman1", "strawman2"]),
    default="confmask",
    help="Algorithm to evaluate.",
)
@click.option("--kr", required=True, type=int, help="Router anonymization degree.")
@click.option("--kh", required=True, type=int, help="Host anonymization degree.")
@click.option("--seed", required=True, type=int, help="Random seed.")
@click.option(
    "--plot-only",
    is_flag=True,
    help="Plot based on stored results without running any evaluation. Ignores -n/--networks.",
)
def main(networks, algorithm, kr, kh, seed, plot_only):
    rich.get_console().rule(f"Figure 5 | {algorithm=}, {kr=}, {kh=}, {seed=}")
    results = {}
    target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
    names = sorted(set(SUPPORTED_NETWORKS) & set(networks)) if not plot_only else []

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
    results_file = RESULTS_DIR / f"5-{target}.json"
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
        plt.figure()
        plt.bar([f"Net{k}" for k, _ in all_results], [v for _, v in all_results])
        plt.ylabel("Avg # of distinct paths between edge routers")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"5-{target}.png")


if __name__ == "__main__":
    main()
