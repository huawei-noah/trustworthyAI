# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def connect(i, G, relation=None) -> set:
    """
    return set of nodes for node i in G with `relation`

    Parameters
    ----------
    i: int
        node
    G: np.ndarray
        a graph matrix
    relation: None or str
        if None, denotes not adjacent

    Returns
    -------
    out: Set
        a set of node or ∅
    """

    from_i = G[i, :] != 0
    to_i = G[:, i] != 0
    if relation is None:
        out = np.where(np.logical_and(~from_i, ~to_i))[0]
    elif relation.lower() == 'neighbors':
        out = np.where(np.logical_and(from_i, to_i))[0]
    elif relation.lower() == 'adjacent':
        out = np.where(np.logical_or(from_i, to_i))[0]
    elif relation.lower() == 'parent':
        out = np.where(np.logical_and(~from_i, to_i))[0]
    elif relation.lower() == 'child':
        out = np.where(np.logical_and(from_i,  ~to_i))[0]
    else:
        raise ValueError(f"The excepted value of parameter `connection` is "
                         f"one of ['neighbors', 'adjacent', None], but got "
                         f"{relation}.")

    return set(out) - {i}


def neighbors(i, G) -> set:
    """return neighbors of node i in graph G"""

    return connect(i, G, relation='neighbors')


def adjacent(i, G) -> set:
    """return adjacent of node i in graph G"""

    return connect(i, G, relation='adjacent')


def parent(i, G) -> set:
    """return parent nodes of i in G"""

    return connect(i, G, relation='parent')


def child(i, G) -> set:
    """return child nodes of i in G"""

    return connect(i, G, relation='child')


def semi_directed_path(x, y, G) -> list:
    """
    Return all paths from x to y in G.
    A semi-directed path between two nodes x and y is a path including both
    directed and undirected edges, but in the case of directed edges,
    they must point to y.

    Parameters
    ----------
    x: int
        the index of the starting node
    y: int
        the index of the target node
    G: np.ndarray
        the adjacency matrix of the graph, where G[i, j] != 0 denotes i -> j.

    Returns
    -------
    semi_paths: list
        all semi-directed paths between x and y
    """

    semi_paths = []
    visitable = {i: child(i, G) | neighbors(i, G) for i in range(G.shape[0])}
    cache = [[x]]
    while len(cache) > 0:
        current_path = cache.pop(0)
        next = list(visitable[current_path[-1]] - set(current_path))
        for next_node in next:
            new_path = current_path.copy()
            new_path.append(next_node)
            if next_node == y:
                semi_paths.append(new_path)
            else:
                cache.append(new_path)

    return semi_paths


def is_clique(sub_nodes, C) -> bool:
    """
    check whether the graph corresponding to the sub-nodes is a complete
    subgraph of graph C

    A subgraph over X is complete if every two nodes in X are connected by some
    edge. The set X is often called a clique;

    Parameters
    ----------
    sub_nodes: set
        sub nodes
    C: np.ndarray
        a graph matrix

    Returns
    -------
    out: bool
        whether sub_nodes is clique
    """

    sub_nodes = list(sub_nodes)
    n = len(sub_nodes)
    subgraph = C[sub_nodes, :][:, sub_nodes]
    sub_skeleton = subgraph + subgraph.T
    edges_num = np.sum((sub_skeleton != 0) * 1)

    return edges_num == n * (n - 1)


def is_dag(G) -> bool:
    """check whether a graph G is DAG"""

    p = np.eye(G.shape[0])
    for _ in range(G.shape[0]):
        p = G @ p
        if np.trace(p) != 0:
            return False

    return True


def pdag_to_cpdag(P) -> np.ndarray:
    """
    transform PDAG to CPDAG

    Parameters
    ----------
    P: np.array
        PDAG

    Returns
    -------
    out: np.array
        CPDAG
    """

    G = pdag_to_dag(P)
    C = dag_to_cpdag(G)

    return C


def pdag_to_dag(P) -> np.ndarray:
    """
    Return an consistent extension of Partially directed acyclic graph (PDAG)

    References
    ----------
    https://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf

    Parameters
    ----------
    P: np.array
        PDAG

    Returns
    -------
    out: np.array
        DAG
    """

    G = only_directed_graph(P)
    all_nodes = list(np.arange(len(P)))
    while (len(P) > 0):
        found = False
        i = 0
        while not found and i < len(P):
            # condition1
            cond1 = (len(child(i, P)) == 0)
            # condition2
            n_i = neighbors(i, P)
            adj_i = adjacent(i, P)
            cond2 = np.all([adj_i - {y} <= adjacent(y, P) for y in n_i])
            if cond1 and cond2:
                found = True
                x = all_nodes[i]
                n_x = [all_nodes[j] for j in n_i]
                G[n_x, x] = 1

                but_x = list(set(range(len(P))) - {i})
                P = P[but_x, :][:, but_x]
                all_nodes.remove(x)
            else:
                i += 1
        if not found:
            raise ValueError("The PDAG does not admit any extension.")

    return G


def dag_to_cpdag(G) -> np.ndarray:
    """
    Return the completed partially directed acyclic graph (CPDAG) that
    represents the Markov equivalence class of a given DAG.

    Parameters
    ----------
    G: np.array
        DAG

    Returns
    -------
    out: np.array
        CPDAG
    """

    labeled_g = label_edges(G)
    cpdag = np.zeros_like(labeled_g)

    compelled = np.argwhere(labeled_g == 1)
    for x, y in compelled:
        cpdag[x, y] = 1
        cpdag[y, x] = 0
    reversible = np.argwhere(labeled_g == 2)
    for x, y in reversible:
        cpdag[x, y], cpdag[y, x] = 1, 1

    return cpdag


def topological_sort(G) -> list:
    """
    return a topological sort of a graph

    Parameters
    ----------
    G: np.ndarray
        must be a DAG

    Returns
    -------
    out: List
        An ordering of the nodes
    """

    if not is_dag(G):
        raise ValueError(f"The input G is not a DAG.")

    ordering = []
    G = G.copy()
    unmarked = list(np.where(G.sum(axis=0) == 0)[0])
    while len(unmarked) > 0:
        x = unmarked.pop()
        ordering.append(x)
        ch_x = child(x, G)
        for y in ch_x:
            G[x, y] = 0   # delete parent
            if len(parent(y, G)) == 0:
                unmarked.append(y)

    return ordering


def order_edges(G) -> tuple:
    """
    produce a total ordering over the edges in a DAG.

    Parameters
    ----------
    G: np.array
        DAG

    Returns
    -------
    out: tuple
        element 0 denotes order_edges list;
        element 1 denotes ordered DAG
    """

    if not is_dag(G):
        raise ValueError(f"The input G is not a DAG.")

    ordered_notes = topological_sort(G)
    ordered_g = (G != 0) * -1
    y_idx = -1
    i = 1
    ordered_edges = []
    while (ordered_g == -1).any():
        y = ordered_notes[y_idx]
        pa_y = parent(y, G)
        for xi in ordered_notes:
            if xi in pa_y:
                ordered_g[xi, y] = i
                ordered_edges.append((xi, y))
                i += 1
        y_idx -= 1

    return ordered_edges, ordered_g


def label_edges(G) -> np.ndarray:
    """
    label edges with 'compelled' or 'reversible'

    Parameters
    ----------
    G: np.array
        DAG

    Returns
    -------
    out: np.array
        DAG with each edge labeled either "compelled" or "reversible"
    """

    ordered_edges, ordered_g = order_edges(G)

    # define: -1: unknown, 1: compelled, 2: reversible,
    labeled_g = (ordered_g != 0) * -1
    while (labeled_g == -1).any():
        lowest_edge = ordered_edges.pop(-1)
        if lowest_edge in np.argwhere(labeled_g == -1):
            x, y = lowest_edge
            goto = False
            w = np.where(labeled_g[:, x] == 1)[0]
            for each_w in w:
                pa_y = parent(y, labeled_g)
                if each_w not in pa_y:
                    labeled_g[x, y] = 1
                    labeled_g[np.where(labeled_g[:, y] == 1)[0], y] = 1
                    goto = True
                    break
                else:
                    labeled_g[each_w, y] = 1
            if not goto:
                pa_x = parent(x, labeled_g)
                z = parent(y, labeled_g)
                if len(z - {x} - pa_x) > 0:
                    labeled_g[x, y] = 1
                    labeled_g[np.where(labeled_g[:, y] == -1)[0], y] = 1
                else:
                    labeled_g[x, y] = 2
                    labeled_g[np.where(labeled_g[:, y] == -1)[0], y] = 2

    return labeled_g


def only_directed_graph(P) -> np.ndarray:
    """
    return a graph contains all of the directed edges from P and no other edges
    """

    G = P.copy()
    G[(G + G.T) == 2] = 0

    return G
