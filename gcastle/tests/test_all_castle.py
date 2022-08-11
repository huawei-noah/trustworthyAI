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
import torch
import logging
import unittest
import traceback
import numpy as np

from castle.algorithms import *
from castle.common import consts

from utils.functional import combined_params


class TestCastleAll(unittest.TestCase):
    """This class for test castle algorithms whether run smoothly. """

    @staticmethod
    def load_data():

        data = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)

        return data

    def add_params(self, params=None) -> dict:
        """
        add parameter `device_type` for algorithms based on torch

        Parameters
        ----------
        params: dict
            from castle.common.consts
        """

        if params is None:
            params = dict()
        if torch.cuda.is_available():
            params['device_type'] = ['cpu', 'gpu']
        else:
            params['device_type'] = ['cpu']

        return params

    def setUp(self) -> None:
        data = self.load_data()
        self.x = data['x']
        self.true_dag = data['y']
        self.rank = np.linalg.matrix_rank(self.true_dag)
        self.error_params = []
        logging.info("Load dataset complete!")

    def tearDown(self) -> None:
        """print which parameter combinations fail to be executed"""

        logging.info(f"{'=' * 20}Test completed!{'=' * 20}")
        logging.info("Failed to execute the following parameter combinations: ")
        if self.error_params:
            for each in self.error_params:
                logging.info(each)

    def test_ANMNonlinear(self):
        """test ANMNonlinear"""

        logging.info(f"{'=' * 20}Start Testing ANMNonlinear{'=' * 20}")
        try:
            algo = ANMNonlinear()
            algo.learn(data=self.x)
        except Exception:
            logging.error(traceback.format_exc())

    def test_CORL(self):
        logging.info(f"{'=' * 20}Start Testing CORL{'=' * 20}")
        params = self.add_params(consts.CORL_VALID_PARAMS)
        for d in combined_params(params):
            try:
                algo = CORL(**d, iteration=3)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_DirectLiNGAM(self) -> None:
        logging.info(f"{'=' * 20}Start Testing DirectLiNGAM{'=' * 20}")
        for d in combined_params(consts.DIRECT_LINGAM_VALID_PARAMS):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = DirectLiNGAM(**d)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_GES_bic_scatter(self) -> None:
        logging.info(f"{'=' * 20}Start Testing GES_bic_scatter{'=' * 20}")
        d = {'criterion': 'bic', 'method': 'scatter'}
        print(f"\n{'=' * 100}")
        print(d)
        print(f"{'=' * 100}")
        try:
            algo = GES(**d)
            algo.learn(data=self.x)
        except Exception:
            self.error_params.append(d)
            print(traceback.format_exc())

    def test_GES_bic_r2(self) -> None:
        logging.info(f"{'=' * 20}Start Testing GES_bic_r2{'=' * 20}")
        d = {'criterion': 'bic', 'method': 'r2'}
        print(f"\n{'=' * 100}")
        print(d)
        print(f"{'=' * 100}")
        try:
            algo = GES(**d)
            algo.learn(data=self.x)
        except Exception:
            self.error_params.append(d)
            print(traceback.format_exc())

    @unittest.skip(reason='Just for discrete data.')
    def test_GES_bdeu(self) -> None:
        logging.info(f"{'=' * 20}Start Testing GES_bdeu{'=' * 20}")
        d = {'criterion': 'bdeu'}
        print(f"\n{'=' * 100}")
        print(d)
        print(f"{'=' * 100}")
        try:
            algo = GES(**d)
            algo.learn(data=self.x)
        except Exception:
            self.error_params.append(d)
            print(traceback.format_exc())

    def test_GOLEM(self) -> None:
        logging.info(f"{'=' * 20}Start Testing GOLEM{'=' * 20}")
        params = self.add_params()
        for d in combined_params(params):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = GOLEM(**d, num_iter=3)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_GraNDAG(self) -> None:
        logging.info(f"{'=' * 20}Start Testing GraNDAG{'=' * 20}")
        params = self.add_params(consts.GRANDAG_VALID_PARAMS)
        for d in combined_params(params):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = GraNDAG(**d, input_dim=self.x.shape[1], iterations=3)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_DAG_GNN(self) -> None:
        logging.info(f"{'=' * 20}Start Testing DAG_GNN{'=' * 20}")
        params = self.add_params(consts.GNN_VALID_PARAMS)
        for d in combined_params(params):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = DAG_GNN(**d, epochs=5, k_max_iter=5)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_ICALiNGAM(self) -> None:
        logging.info(f"{'=' * 20}Start Testing ICALiNGAM{'=' * 20}")
        try:
            # Instantiation algorithm
            algo = ICALiNGAM()
            algo.learn(data=self.x)
        except Exception:
            print(traceback.format_exc())

    def test_Notears(self) -> None:
        logging.info(f"{'=' * 20}Start Testing Notears{'=' * 20}")
        for d in combined_params(consts.NOTEARS_VALID_PARAMS):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = Notears(**d, max_iter=3)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_MCSL(self) -> None:
        logging.info(f"{'=' * 20}Start Testing MCSL{'=' * 20}")
        params = self.add_params(consts.MCSL_VALID_PARAMS)
        for d in combined_params(params):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = MCSL(**d, max_iter=3, iter_step=3)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_NotearsNonlinear(self) -> None:
        logging.info(f"{'=' * 20}Start Testing NotearsNonlinear{'=' * 20}")
        params = self.add_params(consts.NONLINEAR_NOTEARS_VALID_PARAMS)
        for d in combined_params(params):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = NotearsNonlinear(**d, max_iter=3, rho_max=1e4)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_NotearsLowRank(self) -> None:
        logging.info(f"{'=' * 20}Start Testing NotearsLowRank{'=' * 20}")
        try:
            algo = NotearsLowRank(max_iter=3)
            algo.learn(data=self.x, rank=self.rank)
        except Exception:
            print(traceback.format_exc())

    def test_PC(self) -> None:
        logging.info(f"{'=' * 20}Start Testing PC{'=' * 20}")
        for d in combined_params(consts.PC_VALID_PARAMS):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = PC(**d)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())

    def test_RL(self) -> None:
        logging.info(f"{'=' * 20}Start Testing RL{'=' * 20}")
        params = self.add_params(consts.RL_VALID_PARAMS)
        for d in combined_params(params):
            print(f"\n{'=' * 100}")
            print(d)
            print(f"{'=' * 100}")
            try:
                algo = RL(**d, nb_epoch=3)
                algo.learn(data=self.x)
            except Exception:
                self.error_params.append(d)
                print(traceback.format_exc())


if __name__ == '__main__':
    unittest.main()