"""
Utility functions.
"""

from collections import defaultdict

import networkx as nx


def analyze_topology(topology):
    """Obtain information about the underlying graph from network topology.

    Parameters
    ----------
    topology : pd.DataFrame
        Network topology dataframe obtained from Batfish.

    Returns
    -------
    R : set
        Set of routers in the network.
    H : set
        Set of hosts in the network.
    E_R : dict
        Dictionary of router neighbors.
    E_H : dict
        Dictionary of host neighbors.
    E : dict
        Dictionary of all neighbors.
    nx_graph : nx.Graph
        NetworkX graph object.
    """
    E = defaultdict(list)
    for row in topology.itertuples(index=False):
        E[row.Interface.hostname].append(row.Remote_Interface.hostname)
    H = [h for h, neighbors in E.items() if len(neighbors) == 1 and "host" in h]
    R = [r for r, neighbors in E.items() if not (len(neighbors) == 1 and "host" in r)]
    E_R, E_H = {}, {}
    for node, neighbors in E.items():
        if node in H:
            E_H[node] = neighbors
        else:
            E_R[node] = [r for r in neighbors if r in R]

    # Reconstruct the graph
    nx_graph = nx.Graph()
    for i, neighbors in E_R.items():
        for j in neighbors:
            nx_graph.add_edge(i, j)

    return R, H, E_R, E_H, E, nx_graph
