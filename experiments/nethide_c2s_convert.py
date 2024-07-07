import ipaddress
import os
import pickle

# NetHide forwarding graph only contains node name, we need to add interface / prefix information to it
P2P_PREFIXES = list(
    ipaddress.ip_network("192.168.0.0/16").subnets(new_prefix=30)
)  # 2^(30-16) = 16384 subnets
P2P_PREFIX_IDX = -1
ADVERTISE_PREFIXES = list(
    ipaddress.ip_network("10.123.0.0/16").subnets(new_prefix=24)
)  # 2^(24-16) = 256 subnets
ADVERTISE_PREFIX_IDX = -1


def get_next_prefix(typ="p2p"):
    """
    Get the next available prefix based on the type.
    If the type is 'p2p', increment and return the next point-to-point prefix.
    If the type is 'advertise', increment and return the next advertise prefix.
    """
    if typ == "p2p":
        global P2P_PREFIX_IDX
        P2P_PREFIX_IDX += 1
        return P2P_PREFIXES[P2P_PREFIX_IDX]
    elif typ == "advertise":
        global ADVERTISE_PREFIX_IDX
        ADVERTISE_PREFIX_IDX += 1
        return ADVERTISE_PREFIXES[ADVERTISE_PREFIX_IDX]


global SYNTHESIZERS
SYNTHESIZERS = {}
ANOTHER_IF_PENDING_ADD = {}


def create_or_mkdirs(path):
    """
    Create directories if they do not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class RouterSynthesizer:
    def __init__(self, name: str, forwarding_graph: dict) -> None:
        """
        Initialize the RouterSynthesizer with a name and forwarding graph.
        """
        global SYNTHESIZERS
        global ANOTHER_IF_PENDING_ADD
        self.name = name
        self.if_idx = 0
        self.advertise_prefix = self.next_interface(typ="advertise")
        self.neighbors_interfaces = {}
        self.fibs = {}

        self.flush_pending_add()
        self.update_forwarding(forwarding_graph)

    def update_forwarding(self, forwarding_graph: dict):
        """
        Update the forwarding information for the router based on the forwarding graph.
        """
        for dst in forwarding_graph.keys():
            path = forwarding_graph[dst]
            if len(path) == 1:
                continue
            next_hop = path[1]
            route_typ = "OspfIntraAreaRoute"

            self.flush_pending_add()
            if next_hop not in self.neighbors_interfaces and not (
                next_hop in SYNTHESIZERS
                and self.name in SYNTHESIZERS[next_hop].neighbors_interfaces
            ):
                next_interface = self.next_interface()
                hosts = list(next_interface[1].hosts())
                self.neighbors_interfaces[next_hop] = (
                    next_interface[0],
                    hosts[0],
                    next_interface[1],
                )
                if next_hop not in ANOTHER_IF_PENDING_ADD:
                    ANOTHER_IF_PENDING_ADD[next_hop] = []
                ANOTHER_IF_PENDING_ADD[next_hop].append(
                    (self.name, hosts[1], next_interface[1])
                )

            self.fibs[dst] = (self.neighbors_interfaces[next_hop][0], route_typ)

    def flush_pending_add(self):
        """
        Process any pending interface additions.
        """
        for if_info in ANOTHER_IF_PENDING_ADD.get(self.name, []):
            if_name = self.next_interface(new_prefix=False)
            self.neighbors_interfaces[if_info[0]] = (if_name, if_info[1], if_info[2])
        ANOTHER_IF_PENDING_ADD[self.name] = []

    def next_interface(self, typ="p2p", new_prefix=True):
        """
        Generate the next interface name and prefix based on the type.
        """
        self.if_idx += 1
        if not new_prefix:
            return "eth{}".format(self.if_idx)
        return "eth{}".format(self.if_idx), get_next_prefix(typ)

    def gen_c2s_fibs_text(self):
        """
        Generate the C2S FIBs text for the router.
        """
        txt = (
            f"# Router:{self.name}\n"
            + "## VRF:default\n"
            + f"{self.advertise_prefix[1]};{self.advertise_prefix[0]};ConnectedRoute\n"
        )
        for neighbor in self.neighbors_interfaces:
            if_info = self.neighbors_interfaces[neighbor]
            out_if, addr, pfx = self.neighbors_interfaces[neighbor]
            txt += f"{pfx};{out_if};ConnectedRoute\n"
        for dst in self.fibs:
            out_if, route_type = self.fibs[dst]
            dst_pfx = (
                SYNTHESIZERS[dst].advertise_prefix[1]
                if route_type == "OspfIntraAreaRoute"
                else self.neighbors_interfaces[dst][2]
            )
            txt += f"{dst_pfx};{out_if};{route_type}\n"
        return txt


def save_forwarding_to_c2s_fib(forwarding: dict, network_path: str):
    """
    Save the forwarding information to a C2S FIB file.
    """
    for node in forwarding:
        if node not in SYNTHESIZERS:
            router_synthesizer = RouterSynthesizer(node, forwarding[node])
            SYNTHESIZERS[node] = router_synthesizer
        else:
            router_synthesizer = SYNTHESIZERS[node]
            router_synthesizer.update_forwarding(forwarding[node])

    data = {
        "node_neighbors": {},
        "node_networks": {},
    }
    node_c2s_fibs = ""
    for node in forwarding:
        router_synthesizer = SYNTHESIZERS[node]
        router_synthesizer.flush_pending_add()
        data["node_neighbors"][node] = router_synthesizer.neighbors_interfaces
        data["node_networks"][node] = router_synthesizer.advertise_prefix
        node_c2s_fibs += router_synthesizer.gen_c2s_fibs_text()

    create_or_mkdirs(f"{network_path}/fibs")
    pickle.dump(data, open(f"{network_path}/data.pkl", "wb"))
    with open(f"{network_path}/fibs/fib-1.txt", "w+") as f:
        f.write(node_c2s_fibs)
        f.close()
