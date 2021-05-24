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

from tensorflow.keras.layers import Dense, LeakyReLU
from .masked_model import MaskedModel


class MaskedNN(MaskedModel):

    def _forward(self, x):
        for _ in range(self.num_hidden_layers):    # Hidden layer
            x = Dense(self.hidden_size, activation=None, kernel_initializer=self.initializer)(x)
            x = LeakyReLU(alpha=0.05)(x)

        return Dense(1, kernel_initializer=self.initializer)(x)    # Final output layer
