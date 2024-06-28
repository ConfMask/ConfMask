"""
Implementation of the topology obfuscation algorithm in NetHide.
"""

import gurobipy as gp
import networkx as nx
from itertools import combinations, permutations
from Levenshtein import distance


class Network:
    """Network based on an undirected graph.

    Parameters
    ----------
    graph : nx.Graph
        The underlying undirected graph representing the network topology.
    capacity : dict
        The capacity of each egde, i.e., the maximum number of flows that can pass
        through (in either direction) at one time.
    forwarding : dict
        The forwarding behavior of the network. It should be a dictionary mapping each
        node `n` to another dictionary, in which each node `m` is mapped to a list
        representing the forwarding path from `n` to `m` (inclusive of both source and
        destination).
    flows : list, default=None
        The flows in the network. Each flow is a (source, destination) pair, for which
        the forwarding behavior is specified by `forwarding`. If `flows` is None, it is
        considered as all possible (source, destination) pairs in `graph`.
    """

    def __init__(self, graph, capacity, forwarding, flows=None):
        self.graph = graph
        self.capacity = capacity
        self.forwarding = forwarding
        self.flows = flows

        # Determine the actual flows
        if self.flows is None:
            self._flows = list(permutations(self.graph.nodes, 2))
        else:
            self._flows = self.flows

        # Set the capacities and loads of the edges
        nx.set_edge_attributes(self.graph, capacity, "capacity")
        nx.set_edge_attributes(self.graph, self._compute_load(), "load")

    def _compute_load(self):
        """Compute the load on each edge."""
        loads = {edge: 0 for edge in self.graph.edges}
        for source, destination in self._flows:
            if source == destination:
                pass
            route = self.forwarding[source][destination]
            for i in range(len(route) - 1):
                if (route[i], route[i + 1]) in loads:
                    loads[(route[i], route[i + 1])] += 1
                else:
                    loads[(route[i + 1], route[i])] += 1
        return loads

    def flow_accuracy(self, route):
        """Computes the accuracy metric defined in NetHide.

        Parameters
        ----------
        route : list
            A route in the network to compute accuracy of.

        Returns
        -------
        accuracy : float
            The accuracy value.
        """
        assert len(route) != 1
        this_route = self.forwarding[route[0]][route[-1]]
        return 1 - distance(this_route, route) / (len(this_route) + len(route) - 2)

    def flow_utility(self, route):
        """Computes the utility metric defined in NetHide.

        Parameters
        ----------
        route : list
            A route in the network to compute utility of.

        Returns
        -------
        utility : float
            The utility value.
        """
        assert len(route) != 1
        tot_utility = 0
        for i in range(1, len(route)):
            prefix_route = self.forwarding[route[0]][route[i]]
            common = set(prefix_route).intersection(set(route[: (i + 1)]))
            tot_utility += (len(common) - 1) / 2 * (1 / i + 1 / (len(route) - 1))
        return tot_utility / (len(route) - 1)


def _get_candidates(network, n_samples, rng):
    """Sample the candidate forwarding trees that obfuscate the given network.

    This function returns the following data:
    - `trees[n][i]` gives the i-th choice of forwarding tree rooted at node `n`.
    - `accuracies[n][i]` gives the accuracy of `trees[n][i]`.
    - `utilities[n][i]` gives the utility of `trees[n][i]`.
    - `loads[n][i][e]` gives the load that `trees[n][i]` imposes on edge `e`.
    - `all_edges` is the edges of a complete graph constructed from the given network.
    """
    trees, accuracies, utilities, loads = {}, {}, {}, {}

    # Generate a complete graph from the nodes in the original graph
    complete_graph = nx.Graph()
    complete_graph.add_nodes_from(network.graph.nodes)
    complete_graph.add_edges_from(combinations(network.graph.nodes, 2))

    for n in network.graph.nodes:
        trees[n], accuracies[n], utilities[n], loads[n] = [], [], [], []
        for _ in range(n_samples):
            # Sample a subtree rooted at node `n`
            weights = {edge: rng.uniform(1, 10) for edge in network.graph.edges}
            nx.set_edge_attributes(network.graph, values=weights, name="weight")
            _, tree = nx.single_source_dijkstra(network.graph, n)

            sub_accuracy, sub_utility, sub_loads = (
                0,
                0,
                {edge: 0 for edge in complete_graph.edges},
            )
            for route in tree.values():
                if len(route) == 1:
                    continue
                sub_accuracy += network.flow_accuracy(route)
                sub_utility += network.flow_utility(route)
                for j in range(len(route) - 1):
                    if (route[j], route[j + 1]) in sub_loads:
                        sub_loads[(route[j], route[j + 1])] += 1
                    else:
                        sub_loads[(route[j + 1], route[j])] += 1

            trees[n].append(tree)
            accuracies[n].append(sub_accuracy)
            utilities[n].append(sub_utility)
            loads[n].append(sub_loads)
    return trees, accuracies, utilities, loads, complete_graph.edges


def obfuscate(network, default_capacity, rng, n_samples=100, w=0.5):
    """Obfuscates the given network with NetHide.

    Parameters
    ----------
    network : Network
        The network to obfuscate.
    default_capacity : int
        The default capacity of the virtual links that do not exist in the original
        netowkr topology.
    rng : Generator
        The numpy random generator instance.
    n_samples : int, default=100
        The number of forwarding trees to sample for each node.
    w : float, default=0.5
        The weight in [0, 1] of the accuracy metric against the utility metric. In
        particular, the accuracy metric will be assigned weight `w` and the utility
        metric will be assigned weight `1 - w`.

    Returns
    -------
    new_network : Network
        The obfuscated network.
    """
    trees, accuracies, utilities, loads, E = _get_candidates(network, n_samples, rng)
    N, I = network.graph.nodes, range(n_samples)
    c = {e: default_capacity for e in E}
    for e in network.graph.edges:
        c[e] = network.graph.get_edge_date(*e)["capacity"]

    # Initialize the model
    model = gp.Model("NetHide")
    model.params.TimeLimit = 10 * 60

    # Declare decision variables
    take = model.addVars(
        network.graph.nodes, n_samples, vtype=gp.GRB.BINARY, name="Take"
    )

    # Constraint I (deterministic): each node must be assigned exactly one forwarding
    # tree in the virtual topology so as to determine the forwarding behavior
    model.addConstrs((take.sum(n, "*") == 1 for n in N), name="Deterministic")

    # Constraint II (secure): the generated virtual topology must be secure, meaning
    # that the maximum load of each edge must not exceed the corresponding capacity
    model.addConstrs(
        (
            gp.quicksum(loads[n][i][e] * take[n, i] for n in N for i in I) <= c[e]
            for e in E
        ),
        name="Secure",
    )

    # Deploy the objective function: maximize the combination of overall accuracy and
    # overall utility of the chosen virtual topology, represented as the combination of
    # forwarding trees
    model.setObjective(
        gp.quicksum(
            (w * accuracies[n][i] + (1 - w) * utilities[n][i]) * take[n][i]
            for n in N
            for i in I
        ),
        gp.GRB.MAXIMIZE,
    )

    # Perform optimization
    model.optimize()
    forwarding_trees = {}, 0, 0
    for entry in take:
        if take[entry].X == 1:
            forwarding_trees[entry[0]] = trees[entry[0]][entry[1]]

    # Generated the obfuscated network
    new_graph = nx.Graph()
    for tree in forwarding_trees.values():
        for route in tree.values():
            for i in range(len(route) - 1):
                new_graph.add_edge(route[i], route[i + 1])
    return Network(new_graph, c, forwarding_trees)
