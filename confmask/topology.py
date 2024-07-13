"""
Implementation of the topology obfuscation algorithm.
"""

from copy import deepcopy

import numpy as np


def _sorted_degree_sequence(graph):
    """Return the degree sequence and nodes of a graph in descending order."""
    seq = sorted(graph.degree(), key=lambda view: view[1], reverse=True)
    return tuple(map(list, zip(*seq)))


def _sort_wrt_degrees(nodes, degrees):
    """Return sorted nodes and degrees in descending degrees order."""
    nodes, degrees = np.array(nodes), np.array(degrees)
    indices = degrees.argsort()[::-1]
    return nodes[indices].tolist(), degrees[indices].tolist()


def _greedy_examination(degrees, idx, k):
    """The degree examination step in fast k-degree anonymization."""
    n, j = len(degrees), -1

    # Find the first node `j` such that `d(j) < d(idx)`
    for i in range(idx + 1, n):
        if degrees[i] < degrees[idx]:
            j = i
            break

    # If such a node is not found, the degree sequence is already k-anonymous
    if j == -1:
        return n - idx

    this, last = degrees[idx], -1 if idx == 0 else degrees[idx - 1]

    # Case A: If `v(idx)` has the same degree as the last anonymization group, then
    # `v(idx)` to `v(j - 1)` are clustered in an anonymization group and merged into the
    # last anonymization group
    if this == last and n - j >= k:
        return j - idx

    # Case B: This is an exception of case A; if `v(idx)` has the same degree as the
    # last group but there are fewer than `k` nodes starting from `v(j)`, then `v(idx)`
    # to `v(n - 1)` must be clustered into the same group
    if this == last:
        return n - idx

    # Case C: If `v(idx)` does not have the same degree as the last anonymization group,
    # then either `v(idx)` to `v(j - 1)` or `v(idx)` to `v(i + k - 1)` will form a new
    # group
    if n - idx >= 2 * k and n - j >= k:
        return max(k, j - idx)

    # Case D: This is an exception of case C; if `v(idx)` does not have the same degree
    # as the last anonymization group, and further, `v(idx)` to `v(n - 1)` cannot form
    # two groups (having fewer than `2k` nodes), or `v(j)` to `v(n - 1)` cannot form a
    # group (having fewer than `k` nodes), then `v(idx)` to `v(n - 1)` must be clustered
    # into the same group
    return n - idx


def _edge_creation(graph, nodes, degrees, idx, consecutive_count, rng):
    """The edge creation step in fast k-degree anonymization."""
    n, new_edges = len(degrees), []
    for j in range(idx + 1, idx + consecutive_count):
        while degrees[j] < degrees[idx]:
            # Candidates for creating a new edge
            candidates = [
                l
                for l in range(j + 1, n)
                if not graph.has_edge(nodes[j], nodes[l]) and degrees[l] < degrees[idx]
            ]
            if len(candidates) == 0:
                return new_edges, j, nodes, degrees

            # Randomly choose a candidate
            l = rng.choice(candidates)
            degrees[j] += 1
            degrees[l] += 1
            new_edges.append((nodes[j], nodes[l]))
            graph.add_edge(nodes[j], nodes[l])

    nodes, degrees = _sort_wrt_degrees(nodes, degrees)
    return new_edges, -1, nodes, degrees


def _relaxed_edge_creation(graph, nodes, degrees, i, j):
    """The relaxed edge creation step in fast k-degree anonymization."""
    n, new_edges = len(degrees), []
    for l in range(n - 1, -1, -1):
        if not graph.has_edge(nodes[j], nodes[l]) and j != l:
            degrees[j] += 1
            degrees[l] += 1
            new_edges.append((nodes[j], nodes[l]))
            graph.add_edge(nodes[j], nodes[l])
            if degrees[j] == degrees[i]:
                nodes, degrees = _sort_wrt_degrees(nodes, degrees)
                return new_edges, l, nodes, degrees

    assert False, "Logically unreachable"


def fast_k_degree_anonymization(graph, k, rng):
    """Fast k-degree anonymization of graph.

    Parameters
    ----------
    graph : nx.Graph
        The undirected graph to anonymize.
    k : int
        The level of k-degree anonymity to fulfill.
    rng : Generator
        The numpy random generator instance.

    Returns
    -------
    new_graph : nx.Graph
        The anonymized graph.
    new_edges : list
        The edges added to the original graph.

    Notes
    -----
    This algorithm scales up to large graphs. However, it may add more edges than
    necessary for small graph, in which case one should consider using
    `k_degree_anonymization` instead.

    References
    ----------
    [1] Lu, X., Song, Y., Bressan, S. (2012). Fast Identity Anonymization on Graphs. In
        Liddle, S.W., Schewe, KD., Tjoa, A.M., Zhou, X. (eds) Database and Expert
        Systems Applications. DEXA 2012. Lecture Notes in Computer Science, vol 7446.
        Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-32600-4_21
    """
    new_graph, new_edges = deepcopy(graph), []
    nodes, degrees = _sorted_degree_sequence(graph)

    i, step, n = 0, 0, len(degrees)
    assert k <= n
    while i < n:
        group_count = _greedy_examination(degrees, i, k)
        edges, j, nodes, degrees = _edge_creation(
            new_graph, nodes, degrees, i, group_count, rng
        )
        new_edges.extend(edges)

        if j == -1:
            i += group_count
        else:
            edges, l, nodes, degrees = _relaxed_edge_creation(
                new_graph, nodes, degrees, i, j
            )
            new_edges.extend(edges)
            if l < i:
                i = 0
        step += 1

    return new_graph, new_edges


def _anonymization_cost(degrees, k):
    """Compute the anonymization cost of a degree sequence."""
    n = len(degrees)

    # Matrix of costs of anonymization groups, `0 <= i <= j < min(n, i + 2k)`
    I = np.full([n, n], np.inf)
    for i in range(n):
        cur_cost = 0
        for j in range(i, min(i + 2 * k, n)):
            cur_cost += degrees[i] - degrees[j]
            I[i, j] = cur_cost

    # Base case
    DP, anonym_degree_seqs = [np.inf] * n, []
    for i in range(min(2 * k - 1, n)):
        DP[i] = I[0, i]
        anonym_degree_seqs.append([degrees[0]] * (i + 1))

    # Recurrence relation
    for i in range(2 * k - 1, n):
        min_cost, min_degree_seq = I[0, i], [degrees[0]] * (i + 1)
        for t in range(max(k - 1, i - 2 * k), i - k + 1):
            if DP[t] + I[t + 1, i] < min_cost:
                min_cost = DP[t] + I[t + 1, i]
                min_degree_seq = anonym_degree_seqs[t] + [degrees[t + 1]] * (i - t)
        DP[i] = min_cost
        anonym_degree_seqs.append(min_degree_seq)

    return DP[n - 1], anonym_degree_seqs[n - 1]


def _super_graph_realization(graph, sequence, rng):
    """Construct a supergraph of the given graph realizing the given degree sequence."""
    new_graph = graph.copy()
    nodes, degrees = _sorted_degree_sequence(graph)

    # Compute the residual degrees
    residual_degrees = []
    for i in range(len(degrees)):
        if sequence[i] < degrees[i]:
            raise Exception("Degree sequence is unrealizable")
        residual_degrees.append(sequence[i] - degrees[i])
    nodes, residual_degrees = _sort_wrt_degrees(nodes, residual_degrees)

    # Add edges if possible
    new_edges = []
    indices = [i for i in range(len(residual_degrees)) if residual_degrees[i] > 0]
    while len(indices) != 0:
        i = indices.pop(rng.integers(0, len(indices)))
        count, target_count = 0, residual_degrees[i]
        for j in range(len(residual_degrees)):
            # Avoid self-loops and multi-edges
            if (
                residual_degrees[j] >= 1
                and i != j
                and not new_graph.has_edge(nodes[i], nodes[j])
            ):
                new_graph.add_edge(nodes[i], nodes[j])
                new_edges.append((nodes[i], nodes[j]))
                residual_degrees[i] -= 1
                residual_degrees[j] -= 1
                count += 1
                if count == target_count:
                    assert residual_degrees[i] == 0
                    break  # We have created enough new edges for this node
            elif residual_degrees[j] == 0:
                return None, None  # No more possible edges to create

        if count != target_count:
            return None, None  # Failed to create enough new edges

        nodes, residual_degrees = _sort_wrt_degrees(nodes, residual_degrees)
        indices = [i for i in range(len(residual_degrees)) if residual_degrees[i] > 0]

    return new_graph, new_edges


def k_degree_anonymization(graph, k, rng, noise=0.01):
    """K-degree anonymization of graph.

    Parameters
    ----------
    graph : nx.Graph
        The undirected graph to anonymize.
    k : int
        The level of k-degree anonymity to fulfill.
    noise : float
        The proportion of nodes to add noise to if the graph construction algorithm
        fails or the anonymized degree sequence is unrealizable.
    rng : Generator
        The numpy random generator instance.

    Returns
    -------
    new_graph : nx.Graph
        The anonymized graph.
    new_edges : list
        The edges added to the original graph.

    Notes
    -----
    Use `fast_k_degree_anonymization` for large graphs, which scales well and may add
    fewer edges to achieve the desired level of anonymity on large graphs.

    References
    ----------
    [1] Kun Liu and Evimaria Terzi. 2008. Towards Identity Anonymization on Graphs. In
        Proceedings of the 2008 ACM SIGMOD International Conference on Management of
        Data (SIGMOD '08). Association for Computing Machinery, New York, NY, USA, pp.
        93--106. https://doi.org/10.1145/1376616.1376629
    """
    n, step = graph.number_of_nodes(), 0
    assert k <= n

    noise_cnt = min(max(int(noise * n), 1), n)
    nodes, degrees = _sorted_degree_sequence(graph)
    _, anonym_sequence = _anonymization_cost(degrees, k)
    new_graph, new_edges = _super_graph_realization(graph, anonym_sequence, rng)

    # Add noise to the degree sequence and re-anonymize until we find a realization
    while new_graph is None:
        step += 1
        for i in range(noise_cnt):
            if degrees[-i - 1] <= n - 2:
                degrees[-i - 1] += 1
        nodes, degrees = _sort_wrt_degrees(nodes, degrees)
        _, anonym_sequence = _anonymization_cost(degrees, k)
        new_graph, new_edges = _super_graph_realization(graph, anonym_sequence, rng)

    return new_graph, new_edges
