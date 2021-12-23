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
import torch
import torch.nn.functional as F


class DataGenerator(object):
    """Training dataset generator

    Parameters
    ----------
    dataset: array_like
        A 2-dimension np.ndarray
    normalize: bool, default: False
        Whether normalization ``dataset``
    device: option, default: None
        torch.device('cpu') or torch.device('cuda')
    """

    def __init__(self, dataset, normalize=False, device=None) -> None :

        self.dataset = dataset
        self.normalize = normalize
        self.device = device
        self.data_size, self.n_nodes = self.dataset.shape
        self.dataset = torch.tensor(self.dataset,
                                    requires_grad=True,
                                    device=self.device)
        if self.normalize:
            self.dataset = F.normalize(self.dataset)

    def _draw_single_sample(self, dimension) -> torch.Tensor :

        index = np.random.randint(0, self.data_size, size=dimension)
        single_sample = self.dataset[index] # [dimension, n_nodes]

        return single_sample.T # [n_nodes, dimension]

    def draw_batch(self, batch_size, dimension) -> torch.Tensor :
        """Draw batch sample

        Parameters
        ----------
        batch_size: int
            Draw ``batch_size`` single_samples
        dimension: int
            Draw ``dimension`` samples to represent node features
        """

        batch = []
        for _ in range(batch_size):
            single_sample = self._draw_single_sample(dimension=dimension)
            batch.append(single_sample)

        return torch.stack(batch)

