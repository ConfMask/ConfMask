from pathlib import Path

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"

# Mapping from network name to the corresponding protocol; note that all networks used
# in the experiments have OSPF so "bgp" actually means BGP+OSPF
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
CONFMASK_NAME = "confmask-kr{kr}-kh{kh}-seed{seed}"
NETHIDE_NAME = "nethide"

# Level 4: Different subdirectories for storing configurations
ROUTERS_SUBDIR = "configs"
HOSTS_SUBDIR = "hosts"

# File names
STATS_FILE = "_stats.json"
NETHIDE_FORWARDING_FILE = "forwarding.json"
NETHIDE_FORWARDING_ORIGIN_FILE = "forwarding-origin.json"
