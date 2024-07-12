from pathlib import Path

BF_HOST = "localhost"

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"

# All supported algorithms
ALGORITHMS = ["strawman1", "strawman2", "confmask"]
ALGORITHM_LABELS = {
    "strawman1": "Strawman 1",
    "strawman2": "Strawman 2",
    "confmask": "ConfMask",
}

# Mapping from network name to the corresponding protocol; note that all networks used
# in the experiments have OSPF so "bgp" actually means BGP+OSPF
AVAIL_NETWORKS = ["A", "B", "C", "D", "E", "F", "G", "H"]
PROTOCOL_MAPPING = {
    "A": "bgp",
    "B": "bgp",
    "C": "bgp",
    "D": "ospf",
    "E": "ospf",
    "F": "ospf",
    "G": "ospf",
    "H": "ospf",
}

# Level 1: Base directory of all networks
NETWORKS_DIR = Path(__file__).parent.parent / "networks"

# Level 3: Different versions of the same network
ORIGIN_NAME = "origin"
ANONYM_NAME = "{algorithm}-kr{kr}-kh{kh}-seed{seed}"
NETHIDE_NAME = "nethide"

# Level 4: Different subdirectories for storing configurations
ROUTERS_SUBDIR = "configs"
HOSTS_SUBDIR = "hosts"

# File names
STATS_FILE = "_stats.json"
NETHIDE_FORWARDING_FILE = "forwarding.json"
NETHIDE_FORWARDING_ORIGIN_FILE = "forwarding-origin.json"
