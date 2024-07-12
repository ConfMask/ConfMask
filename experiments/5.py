"""
Average number of distinct paths between edge routers in all networks.
"""

import json
from collections import defaultdict
from itertools import permutations

import click
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pybatfish.client.session import Session
from pybatfish.datamodel.flow import HeaderConstraints

import shared
from config import NETWORKS_DIR, RESULTS_DIR, BF_HOST, ANONYM_NAME

bf = Session(host=BF_HOST)


def run_network(network, target, progress, task):
    """Execute the experiment for a single network."""

    def _display(**kwargs):
        progress.update(task, **kwargs)

    _display(description="Uploading configurations...")
    bf.set_network(network)
    bf.init_snapshot(str(NETWORKS_DIR / network / target), name=target, overwrite=True)
    _display(description="Querying topology...")
    topology = bf.q.layer3Edges().answer().frame()
    _display(description="Processing...")

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
                _display(description=f"{host_ips[src_host]} -> {host_ips[dst_host]}...")
                path = trace_route.Traces[0][0]
                paths_mem.add("->".join(hop.node for hop in path.hops[:-1]))

        return len(paths_mem)

    gw_pairs, all_num_paths = list(permutations(gw_nodes, 2)), []
    n_done, n_total = 0, len(gw_pairs)
    _display(details=f"(0/{n_total})")
    for num in Parallel(n_jobs=-1, prefer="threads", return_as="generator_unordered")(
        delayed(_trace)(src_gw, dst_gw) for src_gw, dst_gw in gw_pairs
    ):
        n_done += 1
        all_num_paths.append(num)
        _display(details=f"({n_done}/{n_total})")
    _display(details="")

    result = sum(all_num_paths) / len(all_num_paths)
    _display(description=f"[bold green]Done[/bold green] | {result=:.3f}")
    return result


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
@shared.cli_plot_only()
def main(networks, algorithm, kr, kh, seed, plot_only):
    shared.display_title("Figure 05", algorithm=algorithm, kr=kr, kh=kh, seed=seed)
    results = {}
    target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
    networks = sorted(networks) if not plot_only else []

    def _run_network_func(network, *, progress, task):
        results[network] = run_network(network, target, progress, task)

    shared.display_progress(networks, _run_network_func)

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
