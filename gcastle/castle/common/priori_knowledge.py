# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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


class PrioriKnowledge(object):
    """
    A class for a priori knowledge.

    Parameters
    ----------
    n_nodes: int
        denotes the number of nodes

    Attributes
    ----------
    matrix: np.ndarray
        0  : i does not have a directed path to j;
        1  : i has a directed path to j;
        -1 : No prior background_knowledge is available to know if either of
             the two cases above (0 or 1) is true.
    """

    def __init__(self, n_nodes) -> None:
        self.matrix = (np.zeros((n_nodes, n_nodes), dtype=int)
                       - np.ones((n_nodes, n_nodes), dtype=int)
                       + np.eye(n_nodes, dtype=int))
        self.forbidden_edges = []
        self.required_edges = []

    def add_required_edge(self, i, j) -> None:
        """Add a required edge `i-->j`.

        Parameters
        ----------
        i: int
            denotes location of the source node
        j: int
            denotes location of the target node

        Examples
        --------
        >>> p = PrioriKnowledge(4)
        >>> p.add_required_edge(0, 1)
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [-1  0 -1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        self.matrix[i, j] = 1
        self.required_edges.append((i, j))

    def add_required_edges(self, edges) -> None:
        """Add multiple required edges using a list of tuples.

        Parameters
        ----------
        edges: list
            list of (i, j)

        Examples
        --------
        >>> p = PrioriKnowledge(4)
        >>> p.add_required_edges([(0, 1), (1, 2)]
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [-1  0  1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        for i, j in edges:
            self.add_required_edge(i, j)

    def add_forbidden_edge(self, i, j) -> None:
        """Add a forbidden edge between `i` and `j`.

        Parameters
        ----------
        i: int
            denotes location of the source node
        j: int
            denotes location of the target node

        Examples
        --------
        >>> p = PrioriKnowledge(4)
        >>> p.add_forbidden_edge(0, 1)
        >>> print(p.matrix)
        [[ 0  0 -1 -1]
         [-1  0 -1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        self.matrix[i, j] = 0
        self.forbidden_edges.append((i, j))

    def add_forbidden_edges(self, edges) -> None:
        """Add multiple forbidden edges using a list of tuples.

        Parameters
        ----------
        edges: list
            list of (i, j)

        Examples
        --------
        >>> p = PrioriKnowledge(4)
        >>> p.add_forbidden_edges([(0, 1), (1, 2)]
        >>> print(p.matrix)
        [[ 0  0 -1 -1]
         [-1  0  0 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        for i, j in edges:
            self.add_forbidden_edge(i, j)

    def add_undirected_edge(self, i, j) -> None:
        """Add an edge with unknown direction `i---j`.

        Parameters
        ----------
        i: int
            denotes location of the source node
        j: int
            denotes location of the target node

        Examples
        --------
        >>> p = PrioriKnowledge(4)
        >>> p.add_undirected_edge(0, 1)
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [ 1  0 -1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        self.matrix[i, j] = 1
        self.matrix[j, i] = 1

    def add_undirected_edges(self, edges) -> None:
        """Add multiple edges with unknown direction by using a list of tuples.

        Parameters
        ----------
        edges: list
            list of (i, j)

        Examples
        --------
        >>> p = PrioriKnowledge(4)
        >>> p.add_undirected_edges([(0, 1), (1, 2)]
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [ 1  0  1 -1]
         [-1  1  0 -1]
         [-1 -1 -1  0]]
        """

        for i, j in edges:
            self.add_undirected_edge(i, j)

    def is_forbidden(self, i, j) -> bool:
        return (i, j) in self.forbidden_edges

    def is_required(self, i, j) -> bool:
        return (i, j) in self.required_edges

    def remove_edge(self, i, j) -> None:
        """Remove any edges i---j and j---i."""

        self.matrix[i, j] = 0
        self.matrix[j, i] = 0
        if (i, j) in self.required_edges:
            self.required_edges.remove((i, j))
        if (j, i) in self.required_edges:
            self.required_edges.remove((j, i))
        if (i, j) not in self.forbidden_edges:
            self.forbidden_edges.append((i, j))
        if (j, i) not in self.forbidden_edges:
            self.forbidden_edges.append((j, i))


def orient_by_priori_knowledge(skeleton, priori_knowledge):
    """
    Orient the direction of all edges based on a priori knowledge.

    Parameters
    ----------
    skeleton: np.ndarray
        a skeleton matrix
    priori_knowledge: PrioriKnowledge
        a class object

    Returns
    -------
    out: np.ndarray
        where out[i, j] = out[j, i] = 0 indicates `i   j`;
        out[i, j] = 1 and out[j, i] = 0 indicates `i-->j`;
        out[i, j] = 1 and out[j, i] = 1 indicates `i---j`;
    """

    for i, j in priori_knowledge.required_edges:
        if skeleton[i, j] == 0:
            skeleton[i, j] = 1
    for i, j in priori_knowledge.forbidden_edges:
        if skeleton[i, j] == 1:
            skeleton[i, j] = 0
    return skeleton


if __name__ == '__main__':
    priori = PrioriKnowledge(4)
    priori.add_required_edges([(0, 1), (1, 3)])
    priori.add_forbidden_edges([(1, 2), (2, 3)])
    priori.add_undirected_edge(0, 3)
    print(priori.matrix)
