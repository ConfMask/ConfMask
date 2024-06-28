"""
TODO
"""

import networkx as nx


def generate_gateways(topology):
    """Generate gateways and gateway interfaces from network topology."""
    gws, gw_interfaces = {}, {}
    for _, row in topology.iterrows():
        src_node = row["Interface"].hostname
        if "host" in src_node:
            gws[src_node] = row["Remote_Interface"].hostname
            gw_interfaces[src_node] = row["Remote_Interface"].interface
    return gws, gw_interfaces


def generate_graph(topology):
    """Generate the underlying graph of the network topology."""
    graph = nx.Graph()

    # Bidirectional edges are automatically deduplicated in undirected graphs
    for _, row in topology.items():
        src_node, dst_node = row["Interface"].hostname, row["Remote_Interface"].hostname
        if "host" in src_node or "host" in dst_node:
            continue
        graph.add_edge(src_node, dst_node)
    return graph
