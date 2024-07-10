from pathlib import Path

BF_HOST = "localhost"

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"

# All supported algorithms
ALGORITHMS = ["strawman1", "strawman2", "confmask"]

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
ANONYM_NAME = "{algorithm}-kr{kr}-kh{kh}-seed{seed}"
CONFMASK_NAME = "confmask-kr{kr}-kh{kh}-seed{seed}"  # XXX: replace with ANONYM_NAME
STRAWMAN1_NAME = "strawman1-kr{kr}-kh{kh}-seed{seed}"  # XXX: replace with ANONYM_NAME
STRAWMAN2_NAME = "strawman2-kr{kr}-kh{kh}-seed{seed}"  # XXX: replace with ANONYM_NAME
NETHIDE_NAME = "nethide"

# Level 4: Different subdirectories for storing configurations
ROUTERS_SUBDIR = "configs"
HOSTS_SUBDIR = "hosts"

# File names
STATS_FILE = "_stats.json"
NETHIDE_FORWARDING_FILE = "forwarding.json"
NETHIDE_FORWARDING_ORIGIN_FILE = "forwarding-origin.json"
