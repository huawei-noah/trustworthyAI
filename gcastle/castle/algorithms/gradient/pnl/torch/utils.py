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

import torch
from torch.utils.data import Dataset, DataLoader


class SampleDataset(Dataset):
    """
    construct class for DataLoader

    Parameters
    ----------
    data: sequential array
        if data contains more than one samples set,
        the number of samples in all data must be equal.
    """

    def __init__(self, *data):
        super(SampleDataset, self).__init__()
        if len(set([x.shape[0] for x in data])) != 1:
            raise ValueError("The number of samples in all data must be equal.")
        self.data = data
        self.n_samples = data[0].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        return [d[index] for d in self.data]


def batch_loader(*x, batch_size=64, **kwargs):

    dataset = SampleDataset(*x)
    loader = DataLoader(dataset, batch_size=batch_size, **kwargs)

    return loader


def compute_jacobian(func, inputs):
    """
    Function that computes the Jacobian of a given function.

    See Also
    --------
    torch.autograd.functional.jacobian
    """

    return torch.autograd.functional.jacobian(func, inputs, create_graph=True)


def compute_entropy(x):
    """Computation information entropy of x"""

    distr = torch.distributions.Normal(loc=torch.mean(x),
                                       scale=torch.std(x))
    entropy = distr.entropy()

    return entropy
