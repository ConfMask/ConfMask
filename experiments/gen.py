import ipaddress
import json
import shutil
import time
from collections import defaultdict
from itertools import permutations

import click
import numpy as np
import pandas as pd
import rich
from confmask.ip import generate_unicast_ip, clear_used_ips
from confmask.parser import HostConfigFile, RouterConfigFile, clear_device_ids
from confmask.topology import k_degree_anonymization
from confmask.utils import analyze_topology
from joblib import Parallel, delayed
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session
from pybatfish.datamodel.flow import HeaderConstraints
from pybatfish.datamodel.primitives import Interface

import shared
from config import (
    NETWORKS_DIR,
    ORIGIN_NAME,
    ANONYM_NAME,
    PROTOCOL_MAPPING,
    ROUTERS_SUBDIR,
    HOSTS_SUBDIR,
    STATS_FILE,
    BF_HOST,
)

bf = Session(host=BF_HOST)


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


def _diff_routes(r, host_rib, host_rib_, H_networks):
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


class _Algorithm:
    """The algorithm interface."""

    def __init__(self, network, kr, kh, seed, force_overwrite, progress, task):
        self.network = network
        self.kr = kr
        self.kh = kh
        self.seed = seed
        self.force_overwrite = force_overwrite
        self.progress = progress
        self.task = task

    def _phase(self, description):
        self.progress.update(self.task, description=f"[{self.network}] {description}")

    @property
    def target_name(self):
        """TODO"""
        raise NotImplementedError

    def preprocessing(self):
        """TODO

        Returns
        -------
        skipped : bool
            Whether the task is skipped.
        """
        self.progress.start_task(self.task)

        network_dir = NETWORKS_DIR / self.network
        protocol = PROTOCOL_MAPPING[self.network]
        rng = np.random.default_rng(self.seed)
        start_time = time.perf_counter()

        # Clean up and prepare the target directory
        target_dir = network_dir / self.target_name
        if target_dir.exists() and not self.force_overwrite:
            self._phase("[yellow]Skipped")
            self.progress.stop_task(self.task)
            return True  # Skip the task
        self._phase("Cleaning up target directory...")
        router_config_dir = target_dir / ROUTERS_SUBDIR
        host_config_dir = target_dir / HOSTS_SUBDIR
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        router_config_dir.mkdir()
        host_config_dir.mkdir()

        # Analyze the original network using Batfish
        self._phase("Uploading configurations...")
        bf.set_network(f"{self.network}-{ORIGIN_NAME}")
        bf.init_snapshot(
            str(network_dir / ORIGIN_NAME),
            name=f"{self.network}-{ORIGIN_NAME}",
            overwrite=True,
        )
        self._phase("Querying topology...")
        T = bf.q.layer3Edges().answer().frame()
        self._phase("Querying routes...")
        G = bf.q.routes().answer().frame()
        self._phase("Querying interface properties...")
        LB = bf.q.interfaceProperties().answer().frame()
        self._phase("Processing...")

        # Analyze topology
        R, _, _, E_H, _, nx_graph = analyze_topology(T)

        # Parse router configurations
        R_map = {}  # Maps router name -> (config file path, config object)
        for router_file_path in (network_dir / ORIGIN_NAME / ROUTERS_SUBDIR).iterdir():
            rcf = RouterConfigFile(router_file_path, rng)
            R_map[rcf.name_ori.lower()] = (router_file_path, rcf)

        # Parse host configurations
        H_map = {}  # Maps host name -> (config file path, config object)
        H_networks = []
        for host_file_path in (network_dir / ORIGIN_NAME / HOSTS_SUBDIR).iterdir():
            hcf = HostConfigFile(host_file_path, rng)
            H_map[hcf.name.lower()] = (host_file_path, hcf)
            H_networks.append(hcf.ip_network)

        # Anonymize network topology; NOTE: Prefer `k_degree_anonymization` for smaller
        # graphs and `fast_k_degree_anonymization` for larger graphs
        self._phase("Anonymizing network topology...")
        _, new_edges = k_degree_anonymization(nx_graph, self.kr, rng)

        # For OSPF networks, compute the costs
        self._phase("Computing OSPF costs...")
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
        self._phase("Generating fake interfaces...")
        fake_interfaces = defaultdict(list)  # Maps node -> (interface name, remote IP)

        for u, v in new_edges:
            u_ip_bytes = generate_unicast_ip(rng)
            v_ip_bytes = generate_unicast_ip(
                rng, b0=u_ip_bytes[0], b1=u_ip_bytes[1], b2=u_ip_bytes[2]
            )
            u_ip, v_ip = ".".join(map(str, u_ip_bytes)), ".".join(map(str, v_ip_bytes))

            egde_cost = None if metric_map is None else metric_map[(u, v)]
            u_rcf, v_rcf = R_map[u][1], R_map[v][1]
            u_interface_name = u_rcf.add_interface(u_ip, protocol, egde_cost)
            v_interface_name = v_rcf.add_interface(v_ip, protocol, egde_cost)
            fake_interfaces[u].append((u_interface_name, v_ip))
            fake_interfaces[v].append((v_interface_name, u_ip))
            u_rcf.add_network(u_ip, protocol)
            v_rcf.add_network(v_ip, protocol)

            if (
                protocol == "bgp"
                and u_rcf.has_protocol("bgp")
                and v_rcf.has_protocol("bgp")
            ):
                u_rcf.add_peer(v_ip, v_rcf.bgp_as)
                v_rcf.add_peer(u_ip, u_rcf.bgp_as)

        # Generate fake hosts and connect them to the routers
        self._phase("Generating fake hosts...")
        for _, hcf in H_map.values():
            hcf.generate_fake_hosts(self.kh)
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

        # Write the modified router and host configurations
        self._phase("Writing configurations...")
        for _, rcf in R_map.values():
            rcf.emit(target_dir / ROUTERS_SUBDIR)
        for _, hcf in H_map.values():
            hcf.emit(target_dir / HOSTS_SUBDIR)

        # Store necessary information in the object
        self._phase("Processing...")
        self._target_dir = target_dir
        self._protocol = protocol
        self._rng = rng
        self._start_time = start_time
        self._T = T
        self._G = G
        self._R = R
        self._R_map = R_map
        self._H_networks = H_networks
        self._fake_interfaces = fake_interfaces

        return False

    def fix_routes(self):
        """TODO"""
        raise NotImplementedError

    def output(self, message):
        """TODO"""
        self._phase("Writing configurations...")
        for _, rcf in self._R_map.values():
            rcf.emit(self._target_dir / ROUTERS_SUBDIR)

        end_time = time.perf_counter()
        self._phase(f"{message} | Saving statistics...")
        self.progress.stop_task(self.task)

        # Save statistics
        config_lines_modified = defaultdict(int)
        config_lines_total = defaultdict(int)
        for _, rcf in self._R_map.values():
            lines_count = rcf.count_lines()
            for kind, num in lines_count["modified"].items():
                config_lines_modified[kind] += num
            for kind, num in lines_count["total"].items():
                config_lines_total[kind] += num
        with (self._target_dir / STATS_FILE).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_lines_modified": config_lines_modified,
                    "config_lines_total": config_lines_total,
                    "time_elapsed": end_time - self._start_time,
                },
                f,
                indent=2,
            )
        self._phase(message)

    def run(self):
        """Run the algorithm."""
        skipped = self.preprocessing()
        if skipped:
            return
        message = self.fix_routes()
        self.output(message)


class ConfMask(_Algorithm):
    """The main ConfMask anonymization algorithm."""

    @property
    def target_name(self):
        return ANONYM_NAME.format(
            algorithm="confmask", kr=self.kr, kh=self.kh, seed=self.seed
        )

    def fix_routes(self):
        n_iteration, diff_flag = 0, True
        host_rib, rib_map = _get_host_rib(self._G, self._H_networks, self._phase)

        while diff_flag:
            n_iteration += 1

            self._phase(f"[Iter/{n_iteration}] Uploading configurations...")
            bf.set_network(f"{self.network}-{self.target_name}")
            bf.init_snapshot(
                str(self._target_dir),
                name=f"{self.network}-{self.target_name}",
                overwrite=True,
            )
            self._phase(f"[Iter/{n_iteration}] Querying routes...")
            G_ = bf.q.routes().answer().frame()
            host_rib_, rib_map_ = _get_host_rib(
                G_,
                self._H_networks,
                lambda text: self._phase(f"[Iter/{n_iteration}] {text}"),
            )

            # Compare with original routes
            n_done, n_total = 0, len(self._R)
            self._phase(
                f"[Iter/{n_iteration}] Comparing routes ({n_done}/{n_total})..."
            )
            ospf_set = set()
            for ospf_subset in Parallel(n_jobs=-1, return_as="generator_unordered")(
                delayed(_diff_routes)(r, host_rib, host_rib_, self._H_networks)
                for r in self._R
            ):
                ospf_set |= ospf_subset
                n_done += 1
                self._phase(
                    f"[Iter/{n_iteration}] Comparing routes ({n_done}/{n_total})..."
                )

            # TODO
            if self._protocol == "ospf" and len(ospf_set) > 0:
                self._phase(f"[Iter/{n_iteration}] Incrementing OSPF cost...")
                for node in ospf_set:
                    self._R_map[node][1].incr_ospf_cost()
            else:
                diff_flag, n_total = False, len(self._H_networks)
                for i, h in enumerate(self._H_networks):
                    self._phase(
                        f"[Iter/{n_iteration}] Adjusting routes ({i + 1}/{n_total})..."
                    )
                    h_rib_df = pd.DataFrame(rib_map[h])
                    h_rib_df_ = pd.DataFrame(rib_map_[h])
                    for r in self._R:
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
                                self._R_map[r][1].add_distribute_list(
                                    h, row.Next_Hop, neighbor, row.Protocol
                                )

            # Write the modified router configurations
            self._phase(f"[Iter/{n_iteration}] Writing configurations...")
            for _, rcf in self._R_map.values():
                rcf.emit(self._target_dir / ROUTERS_SUBDIR)

        # Add noises
        n_total = len(self._R)
        for i, r in enumerate(self._R):
            self._phase(f"Adding noise ({i + 1}/{n_total})...")
            for row in G_[G_["Node"] == r].itertuples(index=False):
                if row in host_rib_:
                    continue
                if "bgp" not in row.Protocol and "ospf" not in row.Protocol:
                    continue
                if self._rng.choice([False, True], p=[0.1, 0.9]):
                    continue  # 90% chance skipping

                neighbor = (
                    row.Next_Hop_IP
                    if row.Next_Hop_Interface == "dynamic"
                    and "bgp" in row.Protocol.lower()
                    else None
                )
                self._R_map[r][1].add_distribute_list(
                    ipaddress.ip_network(row.Network),
                    row.Next_Hop,
                    neighbor,
                    row.Protocol,
                )

        return f"[green]Done[/green] in {n_iteration} iterations"


class Strawman1(_Algorithm):
    """Strawman anonymization algorithm 1."""

    @property
    def target_name(self):
        return ANONYM_NAME.format(
            algorithm="strawman1", kr=self.kr, kh=self.kh, seed=self.seed
        )

    def fix_routes(self):
        self._phase("Adding filters...")
        for r in self._R:
            for fake_interface_name, remote_ip in self._fake_interfaces[r]:
                self._R_map[r][1].strawman_add_filter(
                    self._H_networks, fake_interface_name, remote_ip, self._protocol
                )

        return "[green]Done[/green]"


class Strawman2(_Algorithm):
    """Strawman anonymization algorithm 2."""

    @property
    def target_name(self):
        return ANONYM_NAME.format(
            algorithm="strawman2", kr=self.kr, kh=self.kh, seed=self.seed
        )

    def fix_routes(self):
        gw_inf_map = {
            row.Interface.hostname: (row.IPs[0], row.Remote_Interface)
            for row in self._T.itertuples(index=False)
            if "host" in row.Interface.hostname
        }

        def _trace(src_gw, dst_gw):
            """Trace routes between two gateways."""
            start_location = f"@enter({gw_inf_map[src_gw][1]})"
            trace_route = (
                bf.q.traceroute(
                    startLocation=start_location,
                    headers=HeaderConstraints(
                        srcIps=gw_inf_map[src_gw][0],
                        dstIps=gw_inf_map[dst_gw][0],
                    ),
                )
                .answer()
                .frame()
            )

            return src_gw, dst_gw, trace_route.Traces[0]

        # Trace routes of the original network
        gw_pairs = list(permutations(gw_inf_map, 2))
        n_done, n_total = 0, len(gw_pairs)
        origin_traces = defaultdict(dict)
        for src_gw, dst_gw, trace_info in Parallel(
            n_jobs=-1, prefer="threads", return_as="generator_unordered"
        )(delayed(_trace)(src_gw, dst_gw) for src_gw, dst_gw in gw_pairs):
            n_done += 1
            self._phase(f"Tracing original routes ({n_done}/{n_total})...")
            paths_mem = [[hop.node for hop in path.hops[:-1]] for path in trace_info]
            origin_traces[src_gw][dst_gw] = paths_mem

        n_iteration, diff_flag = 0, True
        while diff_flag:
            n_iteration += 1

            self._phase(f"[Iter/{n_iteration}] Uploading configurations...")
            bf.set_network(f"{self.network}-{self.target_name}")
            bf.init_snapshot(
                str(self._target_dir),
                name=f"{self.network}-{self.target_name}",
                overwrite=True,
            )
            self._phase(f"[Iter/{n_iteration}] Querying routes...")
            G_ = bf.q.routes().answer().frame()

            # Compare with original routes
            n_done, n_total = 0, len(self._R)
            self._phase(
                f"[Iter/{n_iteration}] Comparing routes ({n_done}/{n_total})..."
            )
            ospf_set = set()
            for ospf_subset in Parallel(n_jobs=-1, return_as="generator_unordered")(
                delayed(_diff_routes)(r, self._G, G_, self._H_networks) for r in self._R
            ):
                ospf_set |= ospf_subset
                n_done += 1
                self._phase(
                    f"[Iter/{n_iteration}] Comparing routes ({n_done}/{n_total})..."
                )

            if len(ospf_set) == 0:
                break

            # TODO
            if self._protocol == "ospf" and len(ospf_set) > 0:
                self._phase(f"[Iter/{n_iteration}] Incrementing OSPF cost...")
                for node in ospf_set:
                    self._R_map[node][1].incr_ospf_cost()
            else:
                diff_flag, n_total = False, len(self._H_networks)
                n_done, n_total = 0, len(gw_pairs)
                for src_gw, dst_gw, trace_info in Parallel(
                    n_jobs=-1, prefer="threads", return_as="generator_unordered"
                )(delayed(_trace)(src_gw, dst_gw) for src_gw, dst_gw in gw_pairs):
                    n_done += 1
                    self._phase(
                        f"[Iter/{n_iteration}] Tracing/Adjusting routes ({n_done}/{n_total})..."
                    )
                    dst_ip = gw_inf_map[dst_gw][0]
                    for path_info in trace_info:
                        matching_path_found = False
                        deviation = []  # (index in origin_paths_mem, first mismatch)
                        path = [hop.node for hop in path_info.hops[:-1]]
                        origin_paths_mem = origin_traces[src_gw][dst_gw]
                        for idx, origin_path in enumerate(origin_paths_mem):
                            first_mismatch = next(
                                (
                                    i
                                    for i, (hop, origin_hop) in enumerate(
                                        zip(path, origin_path)
                                    )
                                    if hop != origin_hop
                                ),
                                None,
                            )
                            if first_mismatch is None:
                                matching_path_found = True
                                break
                            deviation.append((idx, first_mismatch))

                        if len(deviation) == 0:
                            matching_path_found = True
                        if not matching_path_found:
                            diff_flag = True
                            mx_idx, mx_mismatch = max(deviation, key=lambda x: x[1])

                            # Modify the last matching hop
                            target_r = origin_paths_mem[mx_idx][mx_mismatch - 1]
                            if self._protocol == "ospf":
                                for step in path_info[mx_mismatch - 1].steps:
                                    if step.action == "TRANSMITTED":
                                        next_hop_interface = step.detail.outputInterface
                                        self._R_map[target_r][1].strawman_add_filter(
                                            [ipaddress.ip_network(dst_ip)],
                                            next_hop_interface,
                                            None,
                                            "ospf",
                                        )
                                        break
                            elif self._protocol == "bgp":
                                for step in path_info[mx_mismatch - 1].steps:
                                    if step.action == "FORWARDED":
                                        neighbor = step.detail.routes[0].nextHop.ip
                                        self._R_map[target_r][1].add_distribute_list(
                                            # XXX: Hardcoded because we know our
                                            # networks are /24; generalize if needed
                                            ipaddress.ip_network(
                                                f"{dst_ip}/24", strict=False
                                            ),
                                            None,
                                            neighbor,
                                            "bgp",
                                        )
                                        break

            # Write the modified router configurations
            self._phase(f"[Iter/{n_iteration}] Writing configurations...")
            for _, rcf in self._R_map.values():
                rcf.emit(self._target_dir / ROUTERS_SUBDIR)

        return f"[green]Done[/green] in {n_iteration} iterations"


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
@shared.cli_force_overwrite()
def main(networks, algorithm, kr, kh, seed, force_overwrite):
    rich.get_console().rule(f"Generate | {algorithm=}, {kr=}, {kh=}, {seed=}")
    networks = sorted(networks)

    with Progress(
        TimeElapsedColumn(),
        TextColumn("{task.description}"),
    ) as progress:
        tasks = {
            network: progress.add_task(f"[{network}] (queued)", start=False, total=None)
            for network in networks
        }
        for network in networks:
            clear_device_ids()
            clear_used_ips()
            task = tasks[network]

            try:
                if algorithm == "strawman1":
                    AlgClass = Strawman1
                elif algorithm == "strawman2":
                    AlgClass = Strawman2
                elif algorithm == "confmask":
                    AlgClass = ConfMask
                else:
                    raise NotImplementedError  # unreachable
                alg = AlgClass(network, kr, kh, seed, force_overwrite, progress, task)
                alg.run()

            except Exception:
                # Remove the target directory and print traceback on error
                progress.update(task, description=f"[{network}] [red]Error")
                progress.stop_task(task)
                name = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
                target_dir = NETWORKS_DIR / network / name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                progress.console.print(f"[red]Error in network {network}")
                progress.console.print_exception()


if __name__ == "__main__":
    main()
