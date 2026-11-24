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
import numpy as np
import pandas as pd
from castle.datasets import DAG, THPSimulation, Topology

try:
    from castle.algorithms import TTPM
    TTPM_AVAILABLE = True
except ImportError:
    TTPM_AVAILABLE = False


class TestTHPSimulation(unittest.TestCase):

    def setUp(self) -> None:
        print(f"{'=' * 20}Testing THPSimulation{'=' * 20}")
        np.random.seed(0)
        self.dag = DAG.erdos_renyi(n_nodes=3, n_edges=3, seed=0)
        self.topology_matrix = Topology.erdos_renyi(n_nodes=4, n_edges=4, seed=0)
        # small nodes/horizon with a large mu_range keeps this simulation
        # fast while still producing a non-empty event set
        simulator = THPSimulation(self.dag, self.topology_matrix,
                                  mu_range=(0.05, 0.1),
                                  alpha_range=(0.005, 0.007))
        self.x = simulator.simulate(T=300, max_hop=1, beta=10)

    def test_simulate_keeps_duration_column(self) -> None:
        # regression for trustworthyAI#181: the returned dataframe must
        # retain the 'duration' column produced by the simulation
        self.assertIsInstance(self.x, pd.DataFrame)
        for col in ['node', 'event', 'timestamp', 'duration']:
            self.assertIn(col, self.x.columns)
        self.assertFalse(self.x.empty)
        self.assertTrue(self.x['duration'].notna().all())
        self.assertTrue((self.x['duration'] >= 0).all())

    @unittest.skipUnless(TTPM_AVAILABLE, 'TTPM dependencies are unavailable')
    def test_ttpm_preprocessing_accepts_simulated_data(self) -> None:
        # TTPM only consumes 'event', 'timestamp' and 'node' by name, so the
        # extra 'duration' column must be tolerated by its preprocessing
        ttpm = TTPM(topology_matrix=self.topology_matrix, max_hop=1)
        ttpm._start_init(self.x.copy())
        self.assertEqual(ttpm.tensor.columns.tolist(),
                         ['node', 'timestamp', 'event', 'times'])
        self.assertEqual(ttpm._N, 3)


if __name__ == '__main__':
    unittest.main()
