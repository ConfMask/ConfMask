import ipaddress
import shutil
from collections import defaultdict

import click
import networkx as nx
import numpy as np
import pandas as pd
from confmask.ip import generate_unicast_ip
from confmask.parser import HostConfigFile, RouterConfigFile
from confmask.topology import k_degree_anonymization
from joblib import Parallel, delayed
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session
from pybatfish.datamodel.primitives import Interface

from config import (
    NETWORKS_DIR,
    ORIGIN_NAME,
    CONFMASK_NAME,
    PROTOCOL_MAPPING,
    ROUTERS_SUBDIR,
    HOSTS_SUBDIR,
)


bf = Session(host="localhost")


def _get_host_rib(routes, H_networks, _phase):
    """Get the RIB of the hosts."""
    rows, route_map, n_total = [], defaultdict(list), len(routes)
    for i, row in enumerate(routes.itertuples()):
        dst_network = ipaddress.ip_network(row.Network)
        for network in H_networks:
            if dst_network.supernet_of(network):
                rows.append(row)
                route_map[network].append(row)
        _phase(f"Processing routes ({i + 1}/{n_total})...")
    return pd.DataFrame(rows).drop_duplicates(), route_map


def run_network(network, kr, kh, seed, progress, task):
    """TODO"""
    network_dir = NETWORKS_DIR / network
    protocol = PROTOCOL_MAPPING[network]
    rng = np.random.default_rng([seed, ord(network)])

    def _phase(description):
        progress.update(task, description=f"[{network}] {description}")

    # Analyze the original network using Batfish
    _phase("Uploading configurations...")
    bf.set_network(f"{network}-{ORIGIN_NAME}")
    bf.init_snapshot(
        str(network_dir / ORIGIN_NAME), name=f"{network}-{ORIGIN_NAME}", overwrite=True
    )
    _phase("Querying topology...")
    T = bf.q.layer3Edges().answer().frame()
    _phase("Querying routes...")
    G = bf.q.routes().answer().frame()
    _phase("Querying interface properties...")
    LB = bf.q.interfaceProperties().answer().frame()
    _phase("Processing...")

    # Edges
    E = defaultdict(list)  # Maps node -> neighbors
    for row in T.itertuples(index=False):
        E[row.Interface.hostname].append(row.Remote_Interface.hostname)
    H = set(h for h, neighbors in E.items() if len(neighbors) == 1 and "host" in h)
    R = set(E) - H
    E_R, E_H = {}, {}
    for node, neighbors in E.items():
        if node in H:
            E_H[node] = neighbors
        else:
            E_R[node] = [r for r in neighbors if r in R]

    # Parse router configurations
    R_map = {}  # Maps router name -> (config file path, config object)
    for router_file_path in (network_dir / ORIGIN_NAME / ROUTERS_SUBDIR).iterdir():
        rcf = RouterConfigFile(router_file_path, rng)
        R_map[rcf.name_ori.lower()] = (router_file_path, rcf)

    # Parse host configurations
    H_map = {}  # Maps host name -> (config file path, config object)
    H_networks = []  # Host IP networks
    for host_file_path in (network_dir / ORIGIN_NAME / HOSTS_SUBDIR).iterdir():
        hcf = HostConfigFile(host_file_path, rng)
        H_map[hcf.name.lower()] = (host_file_path, hcf)
        H_networks.append(hcf.ip_network)

    # Anonymize network topology; NOTE: Prefer `k_degree_anonymization` for smaller
    # graphs and `fast_k_degree_anonymization` for larger graphs
    _phase("Anonymizing network topology...")
    nx_graph = nx.Graph()
    for i, neighbors in E_R.items():
        for j in neighbors:
            nx_graph.add_edge(i, j)
    _, new_edges = k_degree_anonymization(nx_graph, kr, rng)

    # For OSPF networks, compute the costs
    _phase("Computing OSPF costs...")
    metric_map = None
    if protocol == "ospf":
        metric_map = {}
        for u, v in new_edges:
            interface_u = Interface(hostname=u, interface=R_map[u][1].interface)
            interface_v = Interface(hostname=v, interface=R_map[v][1].interface)
            lb_u = LB[LB["Interface"] == interface_u].iloc[0]["Primary_Address"]
            lb_v = LB[LB["Interface"] == interface_v].iloc[0]["Primary_Address"]
            network_u = ipaddress.ip_address(lb_u.split("/", 1)[0])
            network_v = ipaddress.ip_address(lb_v.split("/", 1)[0])

            min_cost_u, min_cost_v = np.inf, np.inf
            for row in G[G["Node"] == u].itertuples(index=False):
                if network_v in ipaddress.ip_network(row.Network).hosts():
                    min_cost_u = min(min_cost_u, row.Metric)
            for row in G[G["Node"] == v].itertuples(index=False):
                if network_u in ipaddress.ip_network(row.Network).hosts():
                    min_cost_v = min(min_cost_v, row.Metric)

            assert not np.isinf(min_cost_u) and not np.isinf(min_cost_v)
            metric_map[(u, v)] = max(min_cost_u, min_cost_v)

    # Generate fake interfaces on both ends of each additional edge
    _phase("Generating fake interfaces...")
    fake_interfaces = defaultdict(list)  # Maps node -> (interface name, remote IP)

    def _add_fake(n, ip, edge_cost):
        """Add fake interface and router protocol configurations."""
        rcf = R_map[n][1]
        interface_name = rcf.add_interface(ip, protocol, edge_cost)
        fake_interfaces[n].append((interface_name, ip))
        rcf.add_network(ip, protocol)
        return rcf

    for u, v in new_edges:
        u_ip_bytes = generate_unicast_ip(rng)
        v_ip_bytes = generate_unicast_ip(
            rng, b0=u_ip_bytes[0], b1=u_ip_bytes[1], b2=u_ip_bytes[2]
        )
        u_ip, v_ip = ".".join(map(str, u_ip_bytes)), ".".join(map(str, v_ip_bytes))

        egde_cost = None if metric_map is None else metric_map[(u, v)]
        u_rcf = _add_fake(u, u_ip, egde_cost)
        v_rcf = _add_fake(v, v_ip, egde_cost)

        if (
            protocol == "bgp"
            and u_rcf.has_protocol("bgp")
            and v_rcf.has_protocol("bgp")
        ):
            u_rcf.add_peer(v_ip, v_rcf.bgp_as)
            v_rcf.add_peer(u_ip, u_rcf.bgp_as)

    # Generate fake hosts and connect them to the routers
    _phase("Generating fake hosts...")
    for _, hcf in H_map.values():
        hcf.generate_fake_hosts(kh)
        rcf = R_map[E_H[hcf.name.lower()][0]][1]
        for fake_content in hcf.fake_contents.values():
            rcf.add_interface(
                fake_content["hostInterfaces"]["eth0"]["gateway"], protocol, None
            )
            rcf.add_network(
                fake_content["hostInterfaces"]["eth0"]["gateway"],
                protocol,
                prefix=fake_content["hostInterfaces"]["eth0"]["prefix"],
            )

    # Clean and prepare the target directories
    _phase("Cleaning up target directory...")
    confmask_name = CONFMASK_NAME.format(kr=kr, kh=kh, seed=seed)
    confmask_dir = network_dir / confmask_name
    router_config_dir = confmask_dir / ROUTERS_SUBDIR
    host_config_dir = confmask_dir / HOSTS_SUBDIR
    if confmask_dir.exists():
        shutil.rmtree(confmask_dir)
    confmask_dir.mkdir(parents=True)
    router_config_dir.mkdir()
    host_config_dir.mkdir()

    # Write the modified router and host configurations
    _phase("Writing configurations...")
    for _, rcf in R_map.values():
        rcf.emit(confmask_dir / ROUTERS_SUBDIR)
    for _, hcf in H_map.values():
        hcf.emit(confmask_dir / HOSTS_SUBDIR)

    # Fix the routes
    n_iteration, diff_flag = 0, True
    host_rib, rib_map = _get_host_rib(G, H_networks, _phase)

    while diff_flag:
        n_iteration += 1

        _phase(f"[Iter/{n_iteration}] Uploading configurations...")
        bf.set_network(f"{network}-{confmask_name}")
        bf.init_snapshot(
            str(confmask_dir), name=f"{network}-{confmask_name}", overwrite=True
        )
        _phase(f"[Iter/{n_iteration}] Querying routes...")
        G_ = bf.q.routes().answer().frame()
        host_rib_, rib_map_ = _get_host_rib(
            G_, H_networks, lambda text: _phase(f"[Iter/{n_iteration}] {text}")
        )

        # Compare with original routes
        n_done, n_total = 0, len(R)
        _phase(f"[Iter/{n_iteration}] Comparing routes ({n_done}/{n_total})...")

        def diff_routes(r):
            """Find differences in routes."""
            ospf_subset = set()
            old, new = (
                host_rib[host_rib["Node"] == r],
                host_rib_[host_rib_["Node"] == r],
            )
            for h in H_networks:
                h_rib_old, h_rib_new = [], []  # (node, next hop IP)
                h_nh_old, h_nh_new = set(), set()  # Next hop IPs
                for row in old.itertuples(index=False):
                    dst_network = ipaddress.ip_network(row.Network)
                    if dst_network.supernet_of(h):
                        h_rib_old.append((row.Node, row.Next_Hop_IP))
                        h_nh_old.add(row.Next_Hop_IP)
                for row in new.itertuples(index=False):
                    dst_network = ipaddress.ip_network(row.Network)
                    if dst_network.supernet_of(h):
                        h_rib_new.append((row.Node, row.Next_Hop_IP))
                        h_nh_new.add(row.Next_Hop_IP)

                for node, next_hop in h_rib_old:
                    if next_hop not in h_nh_new:
                        ospf_subset.add(node)

            return ospf_subset

        ospf_set = set()
        for ospf_subset in Parallel(n_jobs=-1, return_as="generator_unordered")(
            delayed(diff_routes)(r) for r in R
        ):
            ospf_set |= ospf_subset
            n_done += 1
            _phase(f"[Iter/{n_iteration}] Comparing routes ({n_done}/{n_total})...")

        # TODO
        if protocol == "ospf" and len(ospf_set) > 0:
            _phase(f"[Iter/{n_iteration}] Incrementing OSPF cost...")
            for node in ospf_set:
                R_map[node][1].incr_ospf_cost()
        else:
            diff_flag, n_total = False, len(H_networks)
            for i, h in enumerate(H_networks):
                _phase(f"[Iter/{n_iteration}] Adjusting routes ({i + 1}/{n_total})...")
                h_rib_df = pd.DataFrame(rib_map[h])
                h_rib_df_ = pd.DataFrame(rib_map_[h])
                for r in R:
                    r_rib_df = h_rib_df[h_rib_df["Node"] == r]
                    r_rib_df_ = h_rib_df_[h_rib_df_["Node"] == r]
                    for row in r_rib_df_.itertuples(index=False):
                        if row.Next_Hop_IP not in r_rib_df["Next_Hop_IP"].values:
                            diff_flag = True
                            neighbor = (
                                row.Next_Hop_IP
                                if row.Next_Hop_Interface == "dynamic"
                                and "bgp" in row.Protocol.lower()
                                else None
                            )
                            R_map[r][1].add_distribute_list(
                                h, row.Next_Hop, neighbor, row.Protocol
                            )

        # Write the modified router configurations
        _phase(f"[Iter/{n_iteration}] Writing configurations...")
        for _, rcf in R_map.values():
            rcf.emit(confmask_dir / ROUTERS_SUBDIR)

    # Add noises
    n_total = len(R)
    for i, r in enumerate(R):
        _phase(f"Adding noise ({i + 1}/{n_total})...")
        for row in G_[G_["Node"] == r].itertuples(index=False):
            if row in host_rib_:
                continue
            if "bgp" not in row.Protocol and "ospf" not in row.Protocol:
                continue
            if rng.choice([False, True], p=[0.1, 0.9]):
                continue  # 90% chance skipping

            neighbor = (
                row.Next_Hop_IP
                if row.Next_Hop_Interface == "dynamic" and "bgp" in row.Protocol.lower()
                else None
            )
            R_map[r][1].add_distribute_list(
                ipaddress.ip_network(row.Network),
                row.Next_Hop,
                neighbor,
                row.Protocol,
            )

    # Write the modified router configurations
    _phase("Writing configurations...")
    for _, rcf in R_map.values():
        rcf.emit(confmask_dir / ROUTERS_SUBDIR)

    _phase(f"[green]Done[/green] in {n_iteration} iterations")
    progress.stop_task(task)


@click.command()
@click.option(
    "-n",
    "--network",
    type=click.Choice(list("ABCDEFGH")),
    required=True,
    help="Network to run.",
)
@click.option("--kr", required=True, type=int, help="Router anonymization degree.")
@click.option("--kh", required=True, type=int, help="Host anonymization degree.")
@click.option("--seed", required=True, type=int, help="Random seed.")
def main(network, kr, kh, seed):
    with Progress(TimeElapsedColumn(), TextColumn("{task.description}")) as progress:
        task = progress.add_task(f"[{network}] Starting...", total=None)
        run_network(network, kr, kh, seed, progress, task)


if __name__ == "__main__":
    main()
