"""
Preserved network specifications via Config2Spec
"""

import json

import click
import glob
import ipaddress
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import subprocess
from confmask.utils import analyze_topology
from rich.progress import Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session

from config import NETWORKS_DIR, CONFMASK_NAME, RESULTS_DIR, ORIGIN_NAME

SUPPORTED_NETWORKS = "A"
CONFIG2SPEC_IMAGE = "ghcr.io/nyu-netsys/confmask-config2spec:latest"
bf = Session(host="localhost")


def run_network(name, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)

    def _phase(description):
        progress.update(task, description=f"[{name}] {description}")

    def _extract_specs(target):
        _read_existed_specs = lambda file: set(open(file, "r").read().splitlines()[1:])
        if os.path.exists(f"{target}/policies.csv"):
            return _read_existed_specs(f"{target}/policies.csv")

        # Extract the network specifications to `policies.csv` with Config2Spec
        subprocess.run([
            "docker", "run", "-it", "--rm", "-v", f"{target}:/snapshot", CONFIG2SPEC_IMAGE, 
            "/opt/config2spec-confmask/ae.sh", "confmask/full-policies", "/snapshot"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _read_existed_specs(f"{target}/policies.csv")

    def _extract_prefixes(path):
        # Extract host prefixes
        host_prefixes = set()
        for h in glob.glob(f"{path}/hosts/host*.json"):
            with open(h) as fp:
                content = json.load(fp)
                host_prefixes.add(ipaddress.ip_network(content['hostInterfaces']['eth0']['prefix'], strict=False))
    
        # Extract config prefixes
        if os.path.exists(f"{path}/config_prefixes.pkl"):
          config_prefixes = pickle.load(open(f"{path}/config_prefixes.pkl", "rb"))

        bf.set_network("test")
        bf.init_snapshot(path, name="test", overwrite=True)

        # Get the prefixes from the config
        config_prefixes = set()
        interfaces = bf.q.interfaceProperties().answer().frame()

        for pfxs in interfaces['All_Prefixes']:
          for pfx in pfxs:
            config_prefixes.add(ipaddress.ip_network(pfx, strict=False))

        pickle.dump(config_prefixes, open(f"{path}/config_prefixes.pkl", "wb"))
        return host_prefixes, config_prefixes
    
    def _find_in_lines(needle, search):
        diff_specs = []
        for i, l in enumerate(search.splitlines() if type(search) is str else search):
            # print(needle.__str__())
            if needle.__str__() in l:
                diff_specs.append(l)
                # print(f"Found {needle} in line {i}:", l)
        return diff_specs

    ORIGIN_SNAPSHOT_PATH = str(NETWORKS_DIR / name / ORIGIN_NAME)
    TARGET_SNAPSHOT_PATH = str(NETWORKS_DIR / name / target)

    # Extract the network specifications
    _phase("[Config2Spec] Extracting origin network specifications...")
    origin_specs = _extract_specs(ORIGIN_SNAPSHOT_PATH)
    _phase("[Config2Spec] Extracting anonymized network specifications...")
    target_specs = _extract_specs(TARGET_SNAPSHOT_PATH)

    # Extract the prefixes to distinguish whether a spec is for a host or for a router interface
    _phase("Extracting prefixes...")
    origin_fakedst_prefixes, origin_config_prefixes = _extract_prefixes(ORIGIN_SNAPSHOT_PATH)
    target_fakedst_prefixes, target_config_prefixes = _extract_prefixes(TARGET_SNAPSHOT_PATH)

    _phase("Calculating...")
    fpos = target_specs - origin_specs
    fneg = origin_specs - target_specs
    kept = origin_specs & target_specs

    fpos_fakedst_count = 0
    fneg_fakedst_count = 0
    fpos_origin_count = 0
    fneg_origin_count = 0
    for pfx in (target_fakedst_prefixes - origin_fakedst_prefixes).union(target_config_prefixes - origin_config_prefixes):
        diff_specs = _find_in_lines(pfx, list(fneg))
        fneg_fakedst_count += len(diff_specs)
        diff_specs = _find_in_lines(pfx, list(fpos))
        fpos_fakedst_count += len(diff_specs)
      
    for pfx in origin_fakedst_prefixes.union(origin_config_prefixes):
        diff_specs = _find_in_lines(pfx, list(fneg))
        fneg_origin_count += len(diff_specs)
        diff_specs = _find_in_lines(pfx, list(fpos))
        fpos_origin_count += len(diff_specs)
    
    result = {
        'origin': {
            'specs': len(origin_specs),
        },
        'target': {
            'specs': len(target_specs),
            'kept': len(kept),
            'fpos': len(fpos),
            'fneg': len(fneg),
            'fpos_host': fpos_fakedst_count,
            'fneg_host': fneg_fakedst_count,
        },
    }

    _phase(f"[green]Done[/green] | ConfMask(kept={len(kept) / len(origin_specs)}, anonym={fpos_fakedst_count / len(origin_specs)}), NetHide(kept=TODO, anonym=TODO)")
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
    results_file = RESULTS_DIR / f"9-{target}.json"
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
        # plt.figure()
        # plt.bar(x, [v for _, (v, _) in all_results], width, label="Original")
        # plt.bar(x + width, [v for _, (_, v) in all_results], width, label="Anonymized")
        # plt.ylabel("Clustering coefficient")
        # plt.ylim(0, 1)
        # plt.xticks(x + width / 2, [f"Net{k}" for k, _ in all_results])
        # plt.legend(loc="upper right")
        # plt.tight_layout()
        # plt.savefig(RESULTS_DIR / f"7-{target}.png")
        # plt.show()


if __name__ == "__main__":
    main()
