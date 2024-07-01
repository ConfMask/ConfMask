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
from experiments.nethide_c2s_convert import save_forwarding_to_c2s_fib
from rich.progress import Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from pybatfish.client.session import Session, HeaderConstraints
from pybatfish.datamodel.flow import PathConstraints

from config import (
    NETWORKS_DIR,
    CONFMASK_NAME,
    RESULTS_DIR,
    ORIGIN_NAME,
    NETHIDE_NAME,
    NETHIDE_FORWARDING_FILE,
    NETHIDE_FORWARDING_ORIGIN_FILE,
    PROTOCOL_MAPPING
)

SUPPORTED_NETWORKS = "ABCDG"
CONFIG2SPEC_IMAGE = "ghcr.io/nyu-netsys/confmask-config2spec:latest"
bf = Session(host="localhost")


def run_network(name, target, progress, task):
    """Execute the experiment for a single network."""
    progress.start_task(task)

    _read_fd = lambda file: json.load((NETWORKS_DIR / name / NETHIDE_NAME / file).open("r", encoding="utf-8"))

    def _phase(description):
        progress.update(task, description=f"[{name}] {description}")

    def _extract_specs(target, mode="full-policies"):
        """
        Extract the network specifications using Config2Spec.

        Args:
            target (str): The target directory.
            mode (str): The mode to run Config2Spec in. Either `confmask/full-policies` or `confmask/nethide-specs`.

        """
        _read_existed_specs = lambda file: set(open(file, "r").read().splitlines()[1:])
        if os.path.exists(f"{target}/policies-{mode}.csv"):
            return _read_existed_specs(f"{target}/policies-{mode}.csv")

        # Extract the network specifications to `policies.csv` with Config2Spec
        # print("Docker command:", " ".join([
        #     "docker", "run", "-it", "--rm", "-v", f"{target}:/snapshot", CONFIG2SPEC_IMAGE, 
        #     "/ae.sh", f"confmask/{mode}", "/snapshot"
        # ]))
        subprocess.run([
            "docker", "run", "-it", "--rm", "-v", f"{target}:/snapshot", CONFIG2SPEC_IMAGE, 
            "/ae.sh", f"confmask/{mode}", "/snapshot"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.rename(f"{target}/policies.csv", f"{target}/policies-{mode}.csv")

        return _read_existed_specs(f"{target}/policies-{mode}.csv")

    def _verify_extracted_specs(target, specs):
        _read_existed_specs = lambda file: set(open(file, "r").read().splitlines())
        if os.path.exists(f"{target}/policies-HOLDS.csv"):
            return _read_existed_specs(f"{target}/policies-HOLDS.csv"), _read_existed_specs(f"{target}/policies-HOLDSNOT.csv")
        bf.set_network("test")
        bf.init_snapshot(target, name="test", overwrite=True)

        def _get_trace_accepted_count(traces):
            accepted_count = 0
            # print(traces[0])
            if len(traces) == 0:
                return 0
            for t in traces[0]:
                if t.disposition in ["ACCEPTED", "EXITS_NETWORK"]:
                    accepted_count += 1
            return accepted_count

        specs_count = len(specs)
        holds_specs = []
        holds_not_specs = []
        for i, spec in enumerate(specs):
            _phase(f"[Config2Spec-PostVerify] Verifying spec {i}/{specs_count}")
            policy_type = spec.split(",")[0]
            src_name = spec.split(",")[3]
            dst_name = spec.split(",")[-2].split(':')[0][1:]
            dst_pfx = spec.split(",")[1]
            result = bf.q.reachability(pathConstraints=PathConstraints(startLocation = src_name), headers=HeaderConstraints(dstIps=dst_pfx, srcIps='0.0.0.0/0'), actions='SUCCESS').answer().frame()

            policy_holds = False

            if policy_type == "PolicyType.Waypoint":
                accepted = _get_trace_accepted_count(result.Traces)
                if accepted > 0:
                    policy_holds = True

            if policy_type == "PolicyType.LoadBalancingSimple":
                accepted = _get_trace_accepted_count(result.Traces)
                if accepted == int(spec.split(",")[2]):
                    policy_holds = True

            if policy_type == "PolicyType.Reachability":
                accepted = _get_trace_accepted_count(result.Traces)
                if accepted > 0:
                    policy_holds = True

            if policy_type == "PolicyType.Isolation":
                accepted = _get_trace_accepted_count(result.Traces)
                if accepted == 0:
                    policy_holds = True

            if policy_holds:
                spec = spec.replace("HOLDSNOT", "HOLDS")
                holds_specs.append(spec)
            else:
                if "HOLDSNOT" not in spec:
                    spec = spec.replace("HOLDS", "HOLDSNOT")
                # print("NotHolds", spec)
                holds_not_specs.append(spec)

        with open(f"{target}/policies-HOLDS.csv", "w") as f:
            f.write("\n".join(holds_specs))

        with open(f"{target}/policies-HOLDSNOT.csv", "w") as f:
            f.write("\n".join(holds_not_specs))
        return set(holds_specs), set(holds_not_specs)

    def _extract_prefixes(path):
        # Extract host prefixes
        host_prefixes = set()
        for h in glob.glob(f"{path}/hosts/host*.json"):
            with open(h) as fp:
                content = json.load(fp)
                host_prefixes.add(ipaddress.ip_network(content['hostInterfaces']['eth0']['prefix'], strict=False))
    
        # Extract config prefixes
        config_prefixes = set()
        if os.path.exists(f"{path}/config_prefixes.pkl"):
          config_prefixes = pickle.load(open(f"{path}/config_prefixes.pkl", "rb"))
        else:
          bf.set_network("test")
          bf.init_snapshot(path, name="test", overwrite=True)

          # Get the prefixes from the config
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

    result = {}

    # Evaluate NetHide: load forwarding graph
    _phase("[NetHide] Loading data...")
    
    # Evaluate NetHide: save forwarding graph to Config2Spec fib-1.txt
    save_forwarding_to_c2s_fib(_read_fd(NETHIDE_FORWARDING_ORIGIN_FILE), ORIGIN_SNAPSHOT_PATH)
    save_forwarding_to_c2s_fib(_read_fd(NETHIDE_FORWARDING_FILE), TARGET_SNAPSHOT_PATH)

    # Evaluate NetHide: extract the network specifications
    _phase("[NetHide] Extracting origin network specifications...")
    nh_origin_specs = _extract_specs(ORIGIN_SNAPSHOT_PATH, "nethide-specs")
    _phase("[NetHide] Extracting anonymized network specifications...")
    nh_target_specs = _extract_specs(TARGET_SNAPSHOT_PATH, "nethide-specs")

    result['nethide'] = {
        'specs_origin': len(nh_origin_specs),
        'specs': len(nh_target_specs),
        'kept': len(nh_origin_specs & nh_target_specs),
        'fpos': len(nh_target_specs - nh_origin_specs),
        'fneg': len(nh_origin_specs - nh_target_specs),

        'kept_ratio': len(nh_origin_specs & nh_target_specs) / len(nh_origin_specs),
        'fpos_ratio': len(nh_target_specs - nh_origin_specs) / len(nh_origin_specs),
        'fneg_ratio': len(nh_origin_specs - nh_target_specs) / len(nh_origin_specs),
    }

    # Evaluate Config2Spec
    # Extract the network specifications
    _phase("[Config2Spec] Extracting origin network specifications...")
    origin_specs = _extract_specs(ORIGIN_SNAPSHOT_PATH)
    _phase("[Config2Spec] Extracting anonymized network specifications...")
    target_specs = _extract_specs(TARGET_SNAPSHOT_PATH)

    # Config2Spec has issues supporting BGP configs that test dataset has, so we use batfish to verify the extracted specs
    if PROTOCOL_MAPPING[name] == "bgp":
        origin_specs, _ = _verify_extracted_specs(ORIGIN_SNAPSHOT_PATH, origin_specs)
        target_specs, _ = _verify_extracted_specs(TARGET_SNAPSHOT_PATH, target_specs)

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
    
    result['origin'] = {
        'specs': len(origin_specs),
    }
    result['confmask'] = {
        'specs': len(target_specs),
        'kept': len(kept),
        'fpos': len(fpos),
        'fneg': len(fneg),
        'fpos_fakedst': fpos_fakedst_count,
        'fneg_host': fneg_fakedst_count,

        'kept_ratio': len(kept) / len(origin_specs),
        'fpos_ratio': len(fpos) / len(origin_specs),
        'fneg_ratio': len(fneg) / len(origin_specs),
        'fpos_fakedst_ratio': fpos_fakedst_count / len(origin_specs),
        'fneg_host_ratio': fneg_fakedst_count / len(origin_specs),
    }

    _phase(f"[green]Done[/green] | ConfMask(kept={result['confmask']['kept_ratio']:.2%}, anonym={result['confmask']['fpos_fakedst_ratio']:.2%}), NetHide(kept={result['nethide']['kept_ratio']:.2%}, fpos={0:.2%})")
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
    all_results.pop("MajorClaims", None)
    all_results.update(results)

    # Extract the data for the plot
    cm_anonym, cm_kept, cm_fneg, cm_fpos, cm_base = [], [], [], [], []
    nh_kept, nh_fpos, nh_fneg, nh_base = [], [], [], []

    for net in all_results:
        # print(all_results[net])
        cm_anonym.append(all_results[net]['confmask']['fpos_fakedst_ratio'])
        cm_kept.append(all_results[net]['confmask']['kept_ratio'])
        cm_fneg.append(all_results[net]['confmask']['fneg_ratio'])
        cm_fpos.append(all_results[net]['confmask']['fpos_ratio'])
        cm_base.append(all_results[net]['origin']['specs'])

        nh_kept.append(all_results[net]['nethide']['kept_ratio'])
        nh_fneg.append(all_results[net]['nethide']['fneg_ratio'])
        nh_fpos.append(all_results[net]['nethide']['fpos_ratio'])
    
    dump_results = all_results.copy()
    dump_results.update({
        "MajorClaims": {
            "ConfMask-Kept": np.mean(cm_kept),
            "ConfMask-Anonym": np.mean(cm_anonym),
            "ConfMask-Fneg": np.mean(cm_fneg),
            "NetHide-Kept": np.mean(nh_kept),
            "NetHide-Fneg": np.mean(nh_fneg),
            "NetHide-Fpos": np.mean(nh_fpos),
            "ConfMask-Reduced-Missing-Specs": (1 - np.mean(cm_fneg) / np.mean(nh_fneg)),
            "ConfMask-Introduced-Anonym-Specs": np.mean(cm_anonym) / np.mean(nh_fpos),
        }
    })

    if not plot_only:
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(dump_results, f, indent=2)

    # Plot the graph
    if len(all_results) > 0:        
        x, width = np.arange(len(all_results)), 0.4
        plt.figure()
        plt.bar(x + width / 2, cm_anonym, width, bottom=np.ones(len(x)), label="ConfMask anonymized")
        plt.bar(x + width / 2, cm_fpos, width, bottom=cm_base, label="ConfMask false positives")
        plt.bar(x + width / 2, cm_fneg, width, bottom=cm_kept, label="ConfMask false negatives")
        plt.bar(x + width / 2, cm_kept, width, label="ConfMask kept specs")

        plt.bar(x - width / 2, nh_fpos, width, bottom=np.ones(len(x)), label="NetHide false positives")
        plt.bar(x - width / 2, nh_fneg, width, bottom=nh_kept, label="NetHide false negatives")
        plt.bar(x - width / 2, nh_kept, width, label="NetHide kept specs")

        plt.ylabel("Specs Difference Ratio")
        plt.ylim(0, 3)
        plt.xticks(x, [f"Net{k}" for k in all_results])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"9-{target}.png")
        # plt.show()


if __name__ == "__main__":
    main()
