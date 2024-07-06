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

from config import (
    NETWORKS_DIR,
    CONFMASK_NAME,
    RESULTS_DIR,
    ORIGIN_NAME,
    NETHIDE_NAME,
    STATS_FILE,
    NETHIDE_FORWARDING_FILE,
    BF_HOST,
)

SUPPORTED_NETWORKS = "ADG"
bf = Session(host=BF_HOST)


def run_network(name, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)

    def _phase(description):
        progress.update(task, description=f"[{name}] {description}")

    # Load entrance interface
    with (NETWORKS_DIR / name / target / STATS_FILE).open("r", encoding="utf-8") as f:
        entrance_interface = json.load(f)["entrance_interface"]

    def _load_trace(ver, prefix):
        """Obtain the forwarding behavior of a network."""
        _phase(f"[{prefix}] Uploading configurations...")
        bf.set_network(name)
        bf.init_snapshot(str(NETWORKS_DIR / name / ver), name=ver, overwrite=True)
        _phase(f"[{prefix}] Quering topology...")
        topology = bf.q.layer3Edges().answer().frame()
        _phase(f"[{prefix}] Processing...")

        # Extract information from the network topology
        gws = {}  # Source host -> destination gateway
        gw_interfaces = {}  # Source host -> destination gateway interface
        router_ips = {}  # Router -> IP
        routers = set()
        for row in topology.itertuples(index=False):
            src_node, dst_node = (
                row.Interface.hostname,
                row.Remote_Interface.hostname,
            )
            if "host" not in src_node and "host" not in dst_node:
                routers.add(src_node)
                routers.add(dst_node)
                router_ips[src_node] = row.IPs[0]
                router_ips[dst_node] = row.Remote_IPs[0]
            elif "host" in src_node:
                gws[src_node] = dst_node
                gw_interfaces[src_node] = row.Remote_Interface.interface

        def _trace(src, dst):
            """Trace routes between two routers."""
            start_location = f"@enter({src}[{entrance_interface[src]}])"
            trace_route = (
                bf.q.traceroute(
                    startLocation=start_location,
                    headers=HeaderConstraints(
                        srcIps=router_ips[src],
                        dstIps=router_ips[dst],
                    ),
                )
                .answer()
                .frame()
            )
            _phase(f"[{prefix}] {router_ips[src]} -> {router_ips[dst]}")
            return src, dst, [hop.node for hop in trace_route.Traces[0][0]]

        router_pairs = list(permutations(routers, 2))
        progress.update(task, total=len(router_pairs), completed=0)

        fd_tree = defaultdict(dict)
        for src, dst, path in Parallel(
            n_jobs=-1, prefer="threads", return_as="generator_unordered"
        )(delayed(_trace)(src, dst) for src, dst in router_pairs):
            fd_tree[src][dst] = path
            progress.update(task, advance=1)

        rich.print(fd_tree)
        rich.print(gws)

        traces = defaultdict(dict)
        for src_host, dst_host in permutations(gws, 2):
            src_gw, dst_gw = gws[src_host], gws[dst_host]
            if src_gw != dst_gw:
                traces[src_host][dst_host] = fd_tree[src_gw][dst_gw]
        return traces, gws

    origin_traces, gws = _load_trace(ORIGIN_NAME, "Original")

    def _compare_with_origin(traces):
        """Get the proportion of exactly kept paths compared with original network."""
        n_same, n_total = 0, 0
        for src_host, all_origin_paths in origin_traces.items():
            for dst_host, origin_path in all_origin_paths.items():
                n_total += 1
                target_path = traces[src_host][dst_host]
                if len(origin_path) == len(target_path):
                    is_same_path = True
                    for origin_hop, target_hop in zip(origin_path, target_path):
                        if origin_hop != target_hop:
                            is_same_path = False
                            break
                    if is_same_path:
                        n_same += 1
        return n_same / n_total

    # Compare ConfMask with the original network
    confmask_traces, _ = _load_trace(target, "ConfMask")
    _phase("[Confmask] Comparing with original network...")
    confmask_prop = _compare_with_origin(confmask_traces)

    # Compare NetHide with the original network
    _phase("[NetHide] Loading data...")
    with (NETWORKS_DIR / name / NETHIDE_NAME / NETHIDE_FORWARDING_FILE).open(
        "r", encoding="utf-8"
    ) as f:
        nethide_fd_tree = json.load(f)
    nethide_traces = defaultdict(dict)
    for src_host, dst_host in permutations(gws, 2):
        src_gw, dst_gw = gws[src_host], gws[dst_host]
        if src_gw != dst_gw:
            nethide_traces[src_host][dst_host] = nethide_fd_tree[src_gw][dst_gw]
    _phase("[NetHide] Comparing with original network...")
    nethide_prop = _compare_with_origin(nethide_traces)

    _phase(
        f"[green]Done[/green] | Confmask: {confmask_prop:.2%} | NetHide: {nethide_prop:.2%}"
    )
    progress.stop_task(task)
    return confmask_prop, nethide_prop


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
@click.option("--kh", required=True, type=int, help="Host anonymization degree.")
@click.option("--seed", required=True, type=int, help="Random seed.")
@click.option(
    "--plot-only",
    is_flag=True,
    help="Plot based on stored results without running any evaluation. Ignores -n/--networks.",
)
def main(networks, kr, kh, seed, plot_only):
    rich.get_console().rule(f"Figure 5 | {kr=}, {kh=}, {seed=}")
    results = {}
    target = CONFMASK_NAME.format(kr=kr, kh=kh, seed=seed)
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
        plt.bar(x + width, [v for _, (v, _) in all_results], width, label="ConfMask")
        plt.ylabel("% Exactly kept paths")
        plt.ylim(0, 1)
        plt.xticks(x + width / 2, [f"Net{k}" for k, _ in all_results])
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"8-{target}.png")


if __name__ == "__main__":
    main()
