"""
Proportion of exactly kept paths.
"""

import json
from collections import defaultdict
from itertools import permutations

import click
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import rich
from joblib import Parallel, delayed
from rich.progress import Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session
from pybatfish.datamodel.flow import HeaderConstraints

import shared
from config import (
    NETWORKS_DIR,
    ANONYM_NAME,
    RESULTS_DIR,
    ORIGIN_NAME,
    NETHIDE_NAME,
    NETHIDE_FORWARDING_FILE,
    BF_HOST,
    ALGORITHM_LABELS,
)

bf = Session(host=BF_HOST)


def run_network(network, algorithm, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)
    network_dir = NETWORKS_DIR / network
    target_label = ALGORITHM_LABELS[algorithm]

    def _phase(description):
        progress.update(task, description=f"[{network}] {description}")

    def _load_fd_tree(ver, prefix):
        """Obtain the forwarding behavior of a network."""
        _phase(f"[{prefix}] Uploading configurations...")
        bf.set_network(network)
        bf.init_snapshot(str(network_dir / ver), name=ver, overwrite=True)
        _phase(f"[{prefix}] Quering topology...")
        topology = bf.q.layer3Edges().answer().frame()
        _phase(f"[{prefix}] Processing...")

        # Extract information from the network topology
        gws = {}  # Source host -> destination gateway
        gw_nodes = defaultdict(list)  # Destination gateway -> source hosts
        gw_interfaces = {}  # Source host -> destination gateway interface
        host_ips = {}  # Host -> IP
        routers = set()  # Set of routers
        router_ips = defaultdict(set)  # Router -> set of IPs
        router_ip2inf = {}  # Router IP -> interface
        for row in topology.itertuples(index=False):
            src_node, dst_node = (
                row.Interface.hostname,
                row.Remote_Interface.hostname,
            )
            if "host" not in src_node and "host" not in dst_node:
                routers.add(src_node)
                routers.add(dst_node)
                router_ips[src_node].add(row.IPs[0])
                router_ips[dst_node].add(row.Remote_IPs[0])
                router_ip2inf[row.IPs[0]] = row.Interface.interface
                router_ip2inf[row.Remote_IPs[0]] = row.Remote_Interface.interface
            elif "host" in src_node:
                gws[src_node] = dst_node
                gw_nodes[dst_node].append(src_node)
                gw_interfaces[src_node] = row.Remote_Interface.interface
                host_ips[src_node] = row.IPs[0]
            elif "host" in dst_node:
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
                    _phase(f"[{prefix}] {host_ips[src_host]} -> {host_ips[dst_host]}")
                    path = trace_route.Traces[0][0]
                    paths_mem.add("->".join(hop.node for hop in path.hops[:-1]))

            for src_ip in router_ips[src_gw]:
                for dst_ip in router_ips[dst_gw]:
                    start_location = f"@enter({src_gw}[{router_ip2inf[src_ip]}])"
                    trace_route = (
                        bf.q.traceroute(
                            startLocation=start_location,
                            headers=HeaderConstraints(srcIps=src_ip, dstIps=dst_ip),
                        )
                        .answer()
                        .frame()
                    )
                    _phase(f"[{prefix}] {src_ip} -> {dst_ip}")
                    path = trace_route.Traces[0][0]
                    paths_mem.add("->".join(hop.node for hop in path.hops))

            return src_gw, dst_gw, paths_mem

        gw_pairs, fd_tree = list(permutations(gw_nodes, 2)), defaultdict(dict)
        progress.update(task, total=len(gw_pairs), completed=0)
        progress._tasks[task].finished_time = None  # Hack to continue elapsed time
        for src_gw, dst_gw, paths_mem in Parallel(
            n_jobs=-1, prefer="threads", return_as="generator_unordered"
        )(delayed(_trace)(src_gw, dst_gw) for src_gw, dst_gw in gw_pairs):
            fd_tree[src_gw][dst_gw] = paths_mem
            progress.update(task, advance=1)

        return fd_tree

    origin_fd_tree = _load_fd_tree(ORIGIN_NAME, "Original")

    def _compare_with_origin(fd_tree):
        """Get the proportion of exactly kept paths compared with original network."""
        n_same, n_total = 0, 0
        for src_gw, all_origin_paths in origin_fd_tree.items():
            for dst_gw, origin_paths in all_origin_paths.items():
                if len(origin_paths & fd_tree[src_gw][dst_gw]) > 0:
                    n_same += 1
                n_total += 1

            # Count the path from source to itself which is always kept (this does not
            # affect ConfMask but makes NetHide a bit happier); it is excluded from the
            # original forwarding tree so we count it here
            n_same += 1
            n_total += 1
        return n_same / n_total

    # Compare ConfMask with the original network
    target_fd_tree = _load_fd_tree(target, target_label)
    _phase(f"[{target_label}] Comparing with original network...")
    target_prop = _compare_with_origin(target_fd_tree)

    # Compare NetHide with the original network
    _phase("[NetHide] Loading data...")
    with (network_dir / NETHIDE_NAME / NETHIDE_FORWARDING_FILE).open(
        "r", encoding="utf-8"
    ) as f:
        nethide_forwarding = json.load(f)
    nethide_fd_tree = defaultdict(dict)
    for src_gw, dst_gw_paths in nethide_forwarding.items():
        for dst_gw, path in dst_gw_paths.items():
            nethide_fd_tree[src_gw][dst_gw] = {"->".join(path)}
    _phase("[NetHide] Comparing with original network...")
    nethide_prop = _compare_with_origin(nethide_fd_tree)

    _phase(
        "[bold green]Done[/bold green]"
        f" | {target_label}: {target_prop:.2%}"
        f" | NetHide: {nethide_prop:.2%}"
    )
    progress.stop_task(task)
    return target_prop, nethide_prop


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
@shared.cli_plot_only()
def main(networks, algorithm, kr, kh, seed, plot_only):
    rich.get_console().rule(f"Figure 8 | {algorithm=}, {kr=}, {kh=}, {seed=}")
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
    results_file = RESULTS_DIR / f"8-{target}.json"
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
        plt.bar(x, [v for _, (_, v) in all_results], width, label="NetHide")
        plt.bar(
            x + width,
            [v for _, (v, _) in all_results],
            width,
            label=ALGORITHM_LABELS[algorithm],
        )
        plt.ylabel("% Exactly kept paths")
        plt.ylim(0, 1)
        plt.xticks(x + width / 2, [f"Net{k}" for k, _ in all_results])
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"8-{target}.png")


if __name__ == "__main__":
    main()
