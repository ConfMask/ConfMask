"""
Preserved network specifications via Config2Spec.
"""

import ipaddress
import json
import pickle
import shutil
import subprocess
from collections import defaultdict
from itertools import count

import click
import matplotlib.pyplot as plt
import numpy as np
import rich
from joblib import Parallel, delayed
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session, HeaderConstraints
from pybatfish.datamodel.flow import PathConstraints

import shared
from config import (
    NETWORKS_DIR,
    ANONYM_NAME,
    RESULTS_DIR,
    ORIGIN_NAME,
    NETHIDE_NAME,
    NETHIDE_FORWARDING_FILE,
    NETHIDE_FORWARDING_ORIGIN_FILE,
    PROTOCOL_MAPPING,
    BF_HOST,
    ALGORITHM_LABELS,
)

bf = Session(host=BF_HOST)

_synthesizers = None
_pending_ifs = None
_prefix_generator = None


def _reset_globals():
    """Reset the global variables for the next network."""
    global _synthesizers, _pending_ifs, _prefix_generator
    _synthesizers = {}  # Maps to router synthesizer instances
    _pending_ifs = defaultdict(list)  # Maps to list of (name, host, prefix)

    # NetHide forwarding graph only contains node names, so we need to attach interface
    # or prefix information to it:
    # - point-to-point 2^(30-16) = 16384 subnets
    # - advertise 2^(24-16) = 256 subnets
    _prefix_generator = {
        "p2p": ipaddress.ip_network("192.168.0.0/16").subnets(new_prefix=30),
        "advertise": ipaddress.ip_network("10.123.0.0/16").subnets(new_prefix=24),
    }


class RouterSynthesizer:
    """Route synthesizer for a router based on its forwarding graph.

    Parameters
    ----------
    name : str
        The name of the router.
    forwarding_graph : dict
        The dictionary mapping each destination node to the full path from to it.
    """

    def __init__(self, name, forwarding_graph):
        self.name = name
        self.forwarding_graph = forwarding_graph

        self._if_idx_generator = count(start=1)
        self._advertise_name = f"eth{next(self._if_idx_generator)}"
        self._advertise_prefix = next(_prefix_generator["advertise"])
        self._neighbors_interfaces = {}  # Maps to (name, host, prefix)
        self._fibs = {}  # Maps to (name, route type)

        self.flush_pending()
        self.update_forwarding(forwarding_graph)

    def update_forwarding(self, forwarding_graph):
        """Update forwarding information based on the given forwarding graph."""
        for dst, path in forwarding_graph.items():
            if len(path) == 1:
                continue
            self.flush_pending()

            next_hop = path[1]
            if next_hop not in self._neighbors_interfaces and not (
                next_hop in _synthesizers
                and self.name in _synthesizers[next_hop].neighbors_interfaces
            ):
                next_if_name = f"eth{next(self._if_idx_generator)}"
                next_if_prefix = next(_prefix_generator["p2p"])
                hosts = next_if_prefix.hosts()

                self._neighbors_interfaces[next_hop] = (
                    next_if_name,
                    next(hosts),
                    next_if_prefix,
                )
                _pending_ifs[next_hop].append((self.name, next(hosts), next_if_prefix))

            self._fibs[dst] = (
                self._neighbors_interfaces[next_hop][0],
                "OspfIntraAreaRoute",
            )

    def flush_pending(self):
        """Process all pending interface additions."""
        for name, host, prefix in _pending_ifs[self.name]:
            self._neighbors_interfaces[name] = (
                f"eth{next(self._if_idx_generator)}",
                host,
                prefix,
            )
        _pending_ifs[self.name].clear()

    def gen_c2s_fibs_lines(self):
        """Generate the Config2Spec FIBs lines for the router."""
        lines = [
            f"# Router:{self.name}\n",
            "## VRF:default\n",
            f"{self._advertise_prefix};{self._advertise_name};ConnectedRoute\n",
        ]

        for neighbor in self._neighbors_interfaces:
            out_if, _, pfx = self._neighbors_interfaces[neighbor]
            lines.append(f"{pfx};{out_if};ConnectedRoute\n")
        for dst in self._fibs:
            out_if, route_type = self._fibs[dst]
            dst_pfx = (
                _synthesizers[dst].advertise_prefix
                if route_type == "OspfIntraAreaRoute"
                else self._neighbors_interfaces[dst][2]
            )
            lines.append(f"{dst_pfx};{out_if};{route_type}\n")
        return lines

    @property
    def neighbors_interfaces(self):
        return self._neighbors_interfaces

    @property
    def advertise_name(self):
        return self._advertise_name

    @property
    def advertise_prefix(self):
        return self._advertise_prefix


def _write_forwarding_info(fd_file, target_dir):
    """Write forwarding information used by Config2Spec."""
    with fd_file.open("r", encoding="utf-8") as f:
        forwarding = json.load(f)

    for node, paths in forwarding.items():
        if node not in _synthesizers:
            router_synthesizer = RouterSynthesizer(node, paths)
            _synthesizers[node] = router_synthesizer
        else:
            router_synthesizer = _synthesizers[node]
            router_synthesizer.update_forwarding(paths)

    data, node_c2s_fibs_lines = {"node_neighbors": {}, "node_networks": {}}, []
    for node in forwarding:
        router_synthesizer = _synthesizers[node]
        router_synthesizer.flush_pending()
        data["node_neighbors"][node] = router_synthesizer.neighbors_interfaces
        data["node_networks"][node] = (
            router_synthesizer.advertise_name,
            router_synthesizer.advertise_prefix,
        )
        node_c2s_fibs_lines.extend(router_synthesizer.gen_c2s_fibs_lines())

    fibs_path = target_dir / "fibs"  # Needed by Config2Spec
    fibs_path.mkdir(parents=True, exist_ok=True)
    with (target_dir / "data.pkl").open("wb") as f:  # Needed Config2Spec/NetHide
        pickle.dump(data, f)
    with (fibs_path / "fib-1.txt").open("w", encoding="utf-8") as f:
        f.writelines(node_c2s_fibs_lines)


def _extract_specs(target, mode):
    """Extract network specifications using Config2Spec.

    Parameters
    ----------
    target : Path
        The target directory.
    mode : {"full-policies", "nethide-specs"}
        The mode to run Config2Spec in. Use "nethide-specs" for NetHide and
        "full-policies" otherwise.
    """
    result = subprocess.run(
        [
            "docker",
            "run",
            "-it",
            "--rm",
            "-v",
            f"{target.as_posix()}:/snapshot",
            "ghcr.io/confmask/confmask-config2spec:latest",
            "/ae.sh",
            f"confmask/{mode}",
            "/snapshot",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    # Read the specifications generated by Config2Spec in the command above
    generated_file = target / "policies.csv"
    if not generated_file.exists():
        raise FileNotFoundError(
            "Config2Spec failed to generate the policies.csv file:\n\n"
            f"Standard output:\n{result.stdout}\n\n"
            f"Standard error:\n{result.stderr}"
        )
    with generated_file.open("r", encoding="utf-8") as f:
        f.readline()  # Skip the csv header
        specs = set(line.rstrip() for line in f)
    generated_file.unlink()
    return specs


def _verify_extracted_specs(target_dir, specs, _display):
    """Post-verification of the extracted specifications using Batfish."""
    bf.set_network("test")
    bf.init_snapshot(str(target_dir), name="test", overwrite=True)

    def _verify_spec(spec):
        """Verify a single specification and return the corrected version."""
        items = spec.split(",")  # 0: type, 1: subnet, 2: specifics, 3: source

        result = bf.q.reachability(
            pathConstraints=PathConstraints(startLocation=items[3]),
            headers=HeaderConstraints(dstIps=items[1], srcIps="0.0.0.0/0"),
            actions="SUCCESS",
        ).answer().frame()

        # Get the number of accepted traces
        if len(result.Traces) > 0:
            accepted_count = sum(
                1
                for trace in result.Traces[0]
                if trace.disposition in ("ACCEPTED", "EXITS_NETWORK")
            )
        else:
            accepted_count = 0

        # Check if the policy holds
        policy_holds = (
            (items[0] == "PolicyType.Waypoint" and accepted_count > 0)
            or (items[0] == "PolicyType.LoadBalancingSimple" and accepted_count == int(items[2]))
            or (items[0] == "PolicyType.Reachability" and accepted_count > 0)
            or (items[0] == "PolicyType.Isolation" and accepted_count == 0)
        )

        if policy_holds:
            return spec.replace("HOLDSNOT", "HOLDS"), True
        elif "HOLDSNOT" not in spec:
            return spec.replace("HOLDS", "HOLDSNOT"), False
        else:
            return spec, False

    holds_specs, holds_not_specs = set(), set()
    n_done, n_total = 0, len(specs)
    for corrected_spec, does_hold in Parallel(
        n_jobs=-1, return_as="generator_unordered"
    )(delayed(_verify_spec)(spec) for spec in specs):
        n_done += 1
        if does_hold:
            holds_specs.add(corrected_spec)
        else:
            holds_not_specs.add(corrected_spec)
        _display(details=f"({n_done}/{n_total})")

    _display(details="")
    return holds_specs, holds_not_specs


def _extract_prefixes(target_dir):
    """Extract host and config prefixes in the snapshot."""
    host_dir, host_prefixes = target_dir / "hosts", set()
    for host_file in host_dir.glob("host*.json"):
        with host_file.open("r", encoding="utf-8") as f:
            host = json.load(f)
        host_prefix = ipaddress.ip_network(
            host["hostInterfaces"]["eth0"]["prefix"], strict=False
        )
        host_prefixes.add(host_prefix)

    # Extract config prefixes
    bf.set_network("test")
    bf.init_snapshot(str(target_dir), name="test", overwrite=True)
    interfaces = bf.q.interfaceProperties().answer().frame()
    config_prefixes = set(
        ipaddress.ip_network(prefix, strict=False)
        for prefixes in interfaces["All_Prefixes"]
        for prefix in prefixes
    )

    return host_prefixes, config_prefixes


def run_network(network, algorithm, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)
    network_dir = NETWORKS_DIR / network
    target_label = ALGORITHM_LABELS[algorithm]

    def _display(**kwargs):
        progress.update(task, **kwargs)

    result = {}
    origin_dir = network_dir / ORIGIN_NAME
    nethide_dir = network_dir / NETHIDE_NAME
    target_dir = network_dir / target

    # Convert forwarding information to Config2Spec format
    _display(description="Loading forwarding info...")
    nethide_dir = network_dir / NETHIDE_NAME
    _write_forwarding_info(nethide_dir / NETHIDE_FORWARDING_ORIGIN_FILE, origin_dir)
    _write_forwarding_info(nethide_dir / NETHIDE_FORWARDING_FILE, target_dir)

    # Evaluate NetHide
    _display(description="[NetHide] Extracting original specs...")
    nh_origin_specs = _extract_specs(origin_dir, "nethide-specs")
    _display(description="[NetHide] Extracting anonymized specs...")
    nh_target_specs = _extract_specs(target_dir, "nethide-specs")

    nh_kept_specs = nh_origin_specs & nh_target_specs
    nh_fpos_specs = nh_target_specs - nh_origin_specs
    nh_fneg_specs = nh_origin_specs - nh_target_specs

    # Record NetHide results
    result["nethide"] = {
        "specs_origin": len(nh_origin_specs),
        "specs_anonym": len(nh_target_specs),
        "kept": len(nh_kept_specs),
        "false_positive": len(nh_fpos_specs),
        "false_negative": len(nh_fneg_specs),
        "kept_ratio": len(nh_kept_specs) / len(nh_origin_specs),
        "false_positive_ratio": len(nh_fpos_specs) / len(nh_origin_specs),
        "false_negative_ratio": len(nh_fneg_specs) / len(nh_origin_specs),
    }

    # Evaluate ConfMask; Config2Spec has issues supporting BGP configurations, in which
    # case we use Batfish to verify the extracted specifications
    _display(description=f"[{target_label}] Extracting original specs...")
    origin_specs = _extract_specs(origin_dir, "full-policies")
    if PROTOCOL_MAPPING[network] == "bgp":
        _display(description=f"[{target_label}] Verifying original specs...")
        origin_specs, _ = _verify_extracted_specs(origin_dir, origin_specs, _display)
    _display(description=f"[{target_label}] Extracting anonymized specs...")
    target_specs = _extract_specs(target_dir, "full-policies")
    if PROTOCOL_MAPPING[network] == "bgp":
        _display(description=f"[{target_label}] Verifying anonymized specs...")
        target_specs, _ = _verify_extracted_specs(target_dir, target_specs, _display)

    # Extract the prefixes to distinguish whether a specification is for a host or for a
    # router interface
    _display(description="Extracting prefixes...")
    origin_fakedst_prefixes, origin_config_prefixes = _extract_prefixes(origin_dir)
    target_fakedst_prefixes, target_config_prefixes = _extract_prefixes(target_dir)

    kept_specs = origin_specs & target_specs
    fpos_specs = target_specs - origin_specs
    fneg_specs = origin_specs - target_specs

    fpos_fakedst_count, fneg_fakedst_count = 0, 0
    fpos_origin_count, fneg_origin_count = 0, 0
    for prefix in (target_fakedst_prefixes - origin_fakedst_prefixes) | (
        target_config_prefixes - origin_config_prefixes
    ):
        fneg_fakedst_count += sum(1 for line in fneg_specs if str(prefix) in line)
        fpos_fakedst_count += sum(1 for line in fpos_specs if str(prefix) in line)
    for prefix in origin_fakedst_prefixes | origin_config_prefixes:
        fneg_origin_count += sum(1 for line in fneg_specs if str(prefix) in line)
        fpos_origin_count += sum(1 for line in fpos_specs if str(prefix) in line)

    result[algorithm] = {
        "specs_origin": len(origin_specs),
        "specs_anonym": len(target_specs),
        # Data
        "kept": len(kept_specs),
        "false_positive": len(fpos_specs),
        "false_negative": len(fneg_specs),
        "false_positive_origin": fpos_origin_count,
        "false_negative_origin": fneg_origin_count,
        "false_positive_fakedst": fpos_fakedst_count,
        "false_negative_fakedst": fneg_fakedst_count,
        # Ratios
        "kept_ratio": len(kept_specs) / len(origin_specs),
        "false_positive_ratio": len(fpos_specs) / len(origin_specs),
        "false_negative_ratio": len(fneg_specs) / len(origin_specs),
        "false_positive_origin_ratio": fpos_origin_count / len(origin_specs),
        "false_negative_origin_ratio": fneg_origin_count / len(origin_specs),
        "false_positive_fakedst_ratio": fpos_fakedst_count / len(origin_specs),
        "false_negative_fakedst_ratio": fneg_fakedst_count / len(origin_specs),
    }

    # Clean up temporary files and directories
    for directory in (origin_dir, target_dir):
        for file in ("acls.txt", "data.pkl", "interfaces.txt", "topology.txt"):
            (directory / file).unlink(missing_ok=True)
        shutil.rmtree(directory / "fibs", ignore_errors=True)

    _display(
        description=(
            "[bold green]Done[/bold green]"
            f" | {target_label}: {result[algorithm]['kept_ratio']:.2%}"
            f" | NetHide: {result['nethide']['kept_ratio']:.2%}"
        )
    )
    progress.stop_task(task)
    return result


@click.command()
@shared.cli_network(multiple=True)
@shared.cli_algorithm()
@shared.cli_kr()
@shared.cli_kh()
@shared.cli_seed()
@shared.cli_plot_only()
def main(networks, algorithm, kr, kh, seed, plot_only):
    rich.get_console().rule(f"Figure 9 | {algorithm=}, {kr=}, {kh=}, {seed=}")
    results = {}
    target = ANONYM_NAME.format(algorithm=algorithm, kr=kr, kh=kh, seed=seed)
    networks = sorted(networks) if not plot_only else []

    if len(networks) > 0:
        with Progress(
            TimeElapsedColumn(),
            TextColumn("{task.description}"),
            TextColumn("{task.fields[details]}", style="dim white"),
        ) as progress:
            tasks = {
                network: progress.add_task(
                    f"[{network}] (queued)", start=False, total=None, details=""
                )
                for network in networks
            }
            for network in networks:
                _reset_globals()
                result = run_network(
                    network, algorithm, target, progress, tasks[network]
                )
                results[network] = result

    # Merge results with existing (if any)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"9-{target}.json"
    all_results = {}
    if results_file.exists():
        with results_file.open("r", encoding="utf-8") as f:
            all_results = json.load(f)
    all_results.pop("MajorClaims", None)
    all_results.update(results)

    # Extract the data for the plot
    cm_anonym, cm_kept, cm_fneg, cm_fpos, cm_base = [], [], [], [], []
    nh_kept, nh_fpos, nh_fneg = [], [], []

    for net in all_results:
        cm_anonym.append(all_results[net][algorithm]["false_positive_fakedst_ratio"])
        cm_kept.append(all_results[net][algorithm]["kept_ratio"])
        cm_fneg.append(all_results[net][algorithm]["false_negative_ratio"])
        cm_fpos.append(all_results[net][algorithm]["false_positive_ratio"])
        cm_base.append(all_results[net][algorithm]["specs_origin"])

        nh_kept.append(all_results[net]["nethide"]["kept_ratio"])
        nh_fneg.append(all_results[net]["nethide"]["false_negative_ratio"])
        nh_fpos.append(all_results[net]["nethide"]["false_positive_ratio"])

    dump_results = all_results.copy()
    dump_results.update(
        {
            "MajorClaims": {
                "ConfMask-Kept": np.mean(cm_kept),
                "ConfMask-Anonym": np.mean(cm_anonym),
                "ConfMask-Fneg": np.mean(cm_fneg),
                "NetHide-Kept": np.mean(nh_kept),
                "NetHide-Fneg": np.mean(nh_fneg),
                "NetHide-Fpos": np.mean(nh_fpos),
                "ConfMask-Reduced-Missing-Specs": (
                    1 - np.mean(cm_fneg) / np.mean(nh_fneg)
                ),
                "ConfMask-Introduced-Anonym-Specs": np.mean(cm_anonym)
                / np.mean(nh_fpos),
            }
        }
    )

    if not plot_only:
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(dump_results, f, indent=2)

    # Plot the graph
    if len(all_results) > 0:
        x, width = np.arange(len(all_results)), 0.4
        plt.figure()
        plt.bar(
            x + width / 2,
            cm_anonym,
            width,
            bottom=np.ones(len(x)),
            label="ConfMask anonymized",
        )
        plt.bar(
            x + width / 2,
            cm_fpos,
            width,
            bottom=cm_base,
            label="ConfMask false positives",
        )
        plt.bar(
            x + width / 2,
            cm_fneg,
            width,
            bottom=cm_kept,
            label="ConfMask false negatives",
        )
        plt.bar(x + width / 2, cm_kept, width, label="ConfMask kept specs")

        plt.bar(
            x - width / 2,
            nh_fpos,
            width,
            bottom=np.ones(len(x)),
            label="NetHide false positives",
        )
        plt.bar(
            x - width / 2,
            nh_fneg,
            width,
            bottom=nh_kept,
            label="NetHide false negatives",
        )
        plt.bar(x - width / 2, nh_kept, width, label="NetHide kept specs")

        plt.ylabel("Specs Difference Ratio")
        plt.ylim(0, 3)
        plt.xticks(x, [f"Net{k}" for k in all_results])
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"9-{target}.png")


if __name__ == "__main__":
    main()
