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

import sys
sys.path.append('../')
import unittest
import traceback
from castle.algorithms import TTPM
from castle.common.consts import TTPM_VALID_PARAMS
from castle.datasets import DAG, THPSimulation, Topology

from utils.functional import combined_params


class TestMCSL(unittest.TestCase):

    def setUp(self) -> None:
        print(f"{'=' * 20}Testing TTPM{'=' * 20}")
        self.dag = DAG.erdos_renyi(n_nodes=10, n_edges=10)
        topology_matrix = Topology.erdos_renyi(n_nodes=20, n_edges=20)
        simulator = THPSimulation(self.dag, topology_matrix,
                                  mu_range=(0.00005, 0.0001),
                                  alpha_range=(0.005, 0.007))
        self.x = simulator.simulate(T=3600 * 24, max_hop=2)
        self.error_params = []

    def tearDown(self) -> None:
        """print which parameter combinations fail to be executed"""
        print(f"{'=' * 20}Test completed!{'=' * 20}")
        print("Failed to execute the following parameter combinations: ")
        if self.error_params:
            for each in self.error_params:
                print(each)

    def test_algo(self) -> None:
        for d in combined_params(TTPM_VALID_PARAMS):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                # Instantiation algorithm
                algo = TTPM(topology_matrix=self.dag, **d, max_iter=3)
                algo.learn(tensor=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())


if __name__ == '__main__':
    unittest.main()
