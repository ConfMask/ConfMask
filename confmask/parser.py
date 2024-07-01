"""
Parser for configuration files.
"""

import json
import ipaddress

from confmask.ip import generate_unicast_ip

_router_ids = set()
_host_ids = set()


def _generate_router_id(rng):
    """Generate a unique router ID."""
    rid = rng.integers(100, 1000, dtype=int)
    while rid in _router_ids:
        rid = rng.integers(100, 1000, dtype=int)
    _router_ids.add(rid)
    return rid


def _generate_host_id(rng):
    """Generate a unique host ID."""
    hid = rng.integers(1000, 10000, dtype=int)
    while hid in _host_ids:
        hid = rng.integers(1000, 10000, dtype=int)
    _host_ids.add(hid)
    return hid


class _Line(str):
    """A line of a router configuration file.

    Use it like a normal string, just with an additional `state` attribute.

    Parameters
    ----------
    state : {0, 1}, default=1
        The state of the line. 0 means the line is original and 1 means the line is
        either modified or newly added.
    """

    def __new__(cls, object, state=1):
        instance = super().__new__(cls, object)
        instance.state = state
        return instance

    def __repr__(self):
        if self.state == 0:
            return super().__repr__()
        return f"\033[31m{super().__repr__()}\033[39m"


class RouterConfigFile:
    """Configuration file for a router.

    Parameters
    ----------
    path : Path
        Path to the configuration file.
    rng : Generator
        The numpy random generator instance.

    Notes
    -----
    The parser is not general-purpose. It is guaranteed to work only with the
    configuration files involved in the experiments.
    """

    def __init__(self, path, rng):
        self.path = path
        self.rng = rng

        self._lines = []
        self._separators = []
        self._fake_interfaces = {}  # Maps interface index to OSPF cost
        self._contents = {
            "prolog": [],  # Blocks of lines before the first interface
            "interface": [],  # Blocks of lines of interface configuration
            "ospf": [],  # Lines of OSPF configuration
            "bgp": [],  # Lines of BGP configuration
            "bgp_address_family": [],  # Lines of BGP address family
            "filter": {},  # Filter name -> filter lines (e.g. access-list)
            "epilog": [],  # Blocks of lines after the protocols
        }

        # Read and parse the configuration file
        with path.open("r", encoding="utf-8") as f:
            segment, separator_flag = "prolog", True
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith("!"):
                    separator_flag = True
                    if segment != "prolog":
                        segment = "epilog"
                else:
                    parts = stripped_line.split(" ", 1)
                    if parts[0] == "hostname":
                        separator_flag = True
                        self._name_ori = parts[1]
                        self._name_reg = f"r{_generate_router_id(rng)}"
                        self._name_idx = len(self._lines)

                    self._lines.append(line)
                    if parts[0] == "interface":
                        segment, separator_flag = "interface", True
                    elif "router ospf" in stripped_line:
                        segment, separator_flag = "ospf", True
                    elif "router bgp" in stripped_line:
                        segment, separator_flag = "bgp", True
                    elif parts[0] == "address-family":
                        segment, separator_flag = "bgp_address_family", True

                    if separator_flag:
                        self._separators.append(len(self._lines))

                    if segment in ("ospf", "bgp"):
                        self._contents[segment].append(_Line(line, state=0))
                    else:
                        if separator_flag:
                            self._contents[segment].append([])
                        self._contents[segment][-1].append(_Line(line, state=0))
                    separator_flag = False

    def add_interface(self, addr, protocol, cost):
        """Add an fake interface to the configuration file.

        Parameters
        ----------
        addr : str
            The IP address.
        protocol : str
            The routing protocol.
        cost : int or None
            The cost for OSPF protocol, otherwise None.

        Returns
        -------
        interface_name : str
            The name of the fake interface.
        """
        self._fake_interfaces[len(self._contents["interface"])] = cost
        interface_name = f"GigabitEthernet{len(self._contents['interface'])}0/0"
        interface_config_lines = [
            _Line(f"interface {interface_name}\n"),
            _Line(f" ip address {addr} 255.255.255.0\n"),
        ]
        if protocol == "ospf" and cost is not None:
            interface_config_lines.append(_Line(" ip ospf cost 1\n"))
        interface_config_lines.append(_Line(" negotiation auto\n"))

        self._contents["interface"].append(interface_config_lines)
        return interface_name

    def add_network(self, addr, protocol, prefix=None):
        """Add network to router the configuration file.

        Parameters
        ----------
        addr : str
            The IP address.
        protocol : str
            The routing protocol.
        prefix : str, optional
            The IP prefix, required for BGP protocol.
        """
        addr_bytes = addr.split(".")
        network_addr = f"{addr_bytes[0]}.{addr_bytes[1]}.{addr_bytes[2]}.0"

        if len(self._contents[protocol]) == 0:
            if len(self._contents["ospf"]) > 0:
                protocol = "ospf"
            elif len(self._contents["bgp"]) > 0:
                protocol = "bgp"

        if protocol == "ospf":
            group = None
            for ospf_line in self._contents["ospf"]:
                if "area" in ospf_line:
                    group = ospf_line.strip().split()[-1]
                    break
            if group is None:
                proc = self._contents["ospf"][0].strip().split()[-1]
                for interface_lines in self._contents["interface"]:
                    for interface_line in interface_lines:
                        if f"ip ospf {proc}" in interface_line:
                            group = interface_line.strip().split()[-1]
                            break
            assert group is not None
            self._contents["ospf"].insert(
                2, _Line(f" network {network_addr} 0.0.0.255 area {group}\n")
            )

        elif protocol == "bgp":
            if len(self._contents["bgp_address_family"]) > 0:
                self._contents["bgp_address_family"][0].insert(
                    1, _Line(f" network {network_addr} mask 255.255.255.0\n")
                )
            else:
                bgp_network_flag, insert_pos = 0, 0
                for i, bgp_line in enumerate(self._contents["bgp"]):
                    if bgp_network_flag == 0:
                        if bgp_line.strip().split(" ", 1)[0] == "network":
                            bgp_network_flag = 1
                    if bgp_network_flag == 1:
                        if bgp_line.strip().split(" ", 1)[0] != "network":
                            bgp_network_flag, insert_pos = 2, i
                            break
                if bgp_network_flag != 2:
                    insert_pos = -1

                if prefix is not None:
                    interface = ipaddress.ip_interface(prefix)
                    bgp_network = interface.network.network_address
                    bgp_mask = interface.netmask
                else:
                    bgp_network = network_addr
                    bgp_mask = "255.255.255.0"
                self._contents["bgp"].insert(
                    insert_pos, _Line(f" network {bgp_network} mask {bgp_mask}\n")
                )

    def add_peer(self, addr, as_number):
        """Add a BGP peer to the configuration file.

        Parameters
        ----------
        addr : str
            The peer IP address.
        as_number : str
            The peer AS number.
        """
        self._contents["bgp"].append(_Line(f" neighbor {addr} remote-as {as_number}\n"))

        if len(self._contents["bgp_address_family"]) > 0:
            self._contents["bgp_address_family"][0].insert(
                -1, _Line(f" neighbor {addr} activate\n")
            )
        else:
            self._contents["bgp"].extend(
                [
                    _Line(f" neighbor {addr} advertisement-interval 0\n"),
                    _Line(f" neighbor {addr} soft-reconfiguration inbound\n"),
                    _Line(f" neighbor {addr} send-community\n"),
                ]
            )

    def add_distribute_list(self, prefix, next_hop, neighbor, protocol):
        """Add distribute list to the configuration file.

        Parameters
        ----------
        prefix : IPv4Network
            The IP prefix.
        next_hop : str
            The next hop IP address.
        neighbor : str or None
            The neighbor IP address.
        protocol : str
            The routing protocol.
        """
        if "ospf" in protocol:
            protocol = "ospf"
        elif "bgp" in protocol:
            protocol = "bgp"
        else:
            assert False, f"{protocol} is not applicable"

        if len(self._contents[protocol]) == 0:
            if len(self._contents["ospf"]) > 0:
                protocol = "ospf"
            elif len(self._contents["bgp"]) > 0:
                protocol = "bgp"

        if protocol == "ospf":
            interface = next_hop.interface
            if interface not in self._contents["filter"]:
                filter_name = f"filter_{len(self._contents['filter']) + 1}"
                self._contents["ospf"].append(
                    _Line(f" distribute-list prefix {filter_name} in {interface}\n")
                )
                self._contents["filter"][interface] = [
                    _Line(f"ip prefix-list {filter_name} permit 0.0.0.0/0 le 32\n")
                ]
            filter_name = self._contents["filter"][interface][-1].strip().split()[2]
            self._contents["filter"][interface].insert(
                0, _Line(f"ip prefix-list {filter_name} deny {prefix}\n")
            )
            return
        elif protocol == "bgp":
            if neighbor not in self._contents["filter"]:
                filter_no = len(self._contents["filter"]) + 1
                self._contents["bgp"].append(
                    _Line(f" neighbor {neighbor} distribute-list {filter_no} in\n")
                )
                self._contents["filter"][neighbor] = [
                    _Line(f"access-list {filter_no} permit any\n")
                ]
            filter_no = self._contents["filter"][neighbor][-1].strip().split()[1]
            self._contents["filter"][neighbor].insert(
                0,
                _Line(
                    f"access-list {filter_no} deny {prefix.network_address} {prefix.hostmask}\n"
                ),
            )

    def incr_ospf_cost(self):
        """Increment OSPF cost of fake interfaces."""
        for i, cost in self._fake_interfaces.items():
            interface_lines = self._contents["interface"][i]
            for j, line in enumerate(interface_lines):
                if "ip ospf cost" in line:
                    orig_cost = int(line.strip().split()[-1])
                    if cost > orig_cost:
                        interface_lines[j] = _Line(f" ip ospf cost {orig_cost + 1}\n")
                    break
            self._contents["interface"][i] = interface_lines

    def has_protocol(self, protocol):
        """Check if the router has the given protocol."""
        return len(self._contents[protocol]) > 0

    def count_modified_lines(self):
        """Count the number of lines modified in the configuration file.

        Returns
        -------
        lines_modified : dict
            The number of modified lines for interface, routing protocols, and filters.
        """
        return {
            "interface": sum(
                1
                for block in self._contents["interface"]
                for line in block
                if line.state == 1
            ),
            "protocol": (
                sum(1 for line in self._contents["ospf"] if line.state == 1)
                + sum(1 for line in self._contents["bgp"] if line.state == 1)
                + sum(
                    1
                    for block in self._contents["bgp_address_family"]
                    for line in block
                    if line.state == 1
                )
            ),
            "filter": sum(
                1
                for block in self._contents["filter"].values()
                for line in block
                if line.state == 1
            ),
        }

    def emit(self, dir):
        """Emit the configuration files to the given directory."""
        lines = []
        for block in self._contents["prolog"]:
            lines.append("!\n")
            lines.extend(block)
        for block in self._contents["interface"]:
            lines.append("!\n")
            lines.extend(block)
        if len(self._contents["ospf"]) > 0:
            lines.append("!\n")
            lines.extend(self._contents["ospf"])
        if len(self._contents["bgp"]) > 0:
            lines.append("!\n")
            lines.extend(self._contents["bgp"])
            if len(self._contents["bgp_address_family"]) > 0:
                for block in self._contents["bgp_address_family"]:
                    lines.append(" !\n")
                    lines.extend(block)
        if len(self._contents["filter"]) > 0:
            for filter_lines in self._contents["filter"].values():
                lines.append("!\n")
                lines.extend(filter_lines)
        for block in self._contents["epilog"]:
            lines.append("!\n")
            lines.extend(block)

        with (dir / f"{self._name_ori}.cfg").open("w", encoding="utf-8") as f:
            f.writelines(lines)

    @property
    def name_ori(self):
        return self._name_ori

    @property
    def interface(self):
        return self._contents["interface"][0][0].strip().split(" ", 1)[1]

    @property
    def bgp_as(self):
        return self._contents["bgp"][0].strip().split()[2]


class HostConfigFile:
    """Configuration file for a host.

    Parameters
    ----------
    path : Path
        Path to the configuration file.
    rng : Generator
        The numpy random generator instance.

    Notes
    -----
    The parser is not general-purpose. It is guaranteed to work only with the
    configuration files involved in the experiments.
    """

    def __init__(self, path, rng):
        self.path = path
        self.rng = rng

        self._id = _generate_host_id(rng)
        with path.open("r", encoding="utf-8") as f:
            self._contents = json.load(f)
        self._fake_contents = {}  # Fake host ID -> contents

    def generate_fake_hosts(self, k):
        """Generate k-1 fake hosts."""
        for _ in range(k - 1):
            fake_ip_bytes = generate_unicast_ip(self.rng, b0=32, b3=101)
            fake_ip = ".".join(map(str, fake_ip_bytes))
            fake_id = _generate_host_id(self.rng)
            self._fake_contents[fake_id] = {
                "hostname": f"host{fake_id}",
                "hostInterfaces": {
                    "eth0": {
                        "name": "eth0",
                        "prefix": f"{fake_ip}/24",
                        "gateway": f"32.{fake_ip_bytes[1]}.{fake_ip_bytes[2]}.1",
                    }
                },
            }

    def emit(self, dir):
        """Emit the configuration files to the given directory."""
        # Real host
        new_contents = self._contents.copy()
        new_contents["hostname"] = f"host{self._id}"
        with (dir / f"host{self._id}.json").open("w", encoding="utf-8") as f:
            json.dump(self._contents, f, indent=2)

        # Fake hosts
        for fake_id, fake_content in self._fake_contents.items():
            with (dir / f"host{fake_id}.json").open("w", encoding="utf-8") as f:
                json.dump(fake_content, f, indent=2)

    def emit_fakes(self, dir):
        """Emit the fake configuration files to the given directory."""
        pass  # TODO

    @property
    def name(self):
        return self._contents["hostname"]

    @property
    def fake_contents(self):
        return self._fake_contents

    @property
    def ip_network(self):
        return ipaddress.ip_network(
            self._contents["hostInterfaces"]["eth0"]["prefix"], strict=False
        )
