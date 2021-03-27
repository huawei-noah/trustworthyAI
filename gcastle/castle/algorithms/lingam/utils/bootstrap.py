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

import numbers
import numpy as np
from sklearn.utils import check_array, resample


class BootstrapMixin():
    """Mixin class for all LiNGAM algorithms that implement the method of bootstrapping."""

    def bootstrap(self, X, n_sampling):
        """
        Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : BootstrapResult
            Returns the result of bootstrapping.
        """
        # Check parameters
        X = check_array(X)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError(
                    'n_sampling must be an integer greater than 0.')
        else:
            raise ValueError('n_sampling must be an integer greater than 0.')

        # Bootstrapping
        adjacency_matrices = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        total_effects = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        for i in range(n_sampling):
            self.fit(resample(X))
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c+1:]:
                    total_effects[i, to, from_] = self.estimate_total_effect(
                        X, from_, to)

        return BootstrapResult(adjacency_matrices, total_effects)


class BootstrapResult(object):
    """The result of bootstrapping."""

    def __init__(self, adjacency_matrices, total_effects):
        """
        Construct a BootstrapResult.

        Parameters
        ----------
        adjacency_matrices : array-like, shape (n_sampling)
            The adjacency matrix list by bootstrapping.
        total_effects : array-like, shape (n_sampling)
            The total effects list by bootstrapping.
        """
        self._adjacency_matrices = adjacency_matrices
        self._total_effects = total_effects

    @property
    def adjacency_matrices_(self):
        """
        The adjacency matrix list by bootstrapping.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (n_sampling)
            The adjacency matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._adjacency_matrices

    @property
    def total_effects_(self):
        """
        The total effect list by bootstrapping.

        Returns
        -------
        total_effects_ : array-like, shape (n_sampling)
            The total effect list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._total_effects

    def get_causal_direction_counts(self, n_directions=None, min_causal_effect=None, split_by_causal_effect_sign=False):
        """
        Get causal direction count as a result of bootstrapping.

        Parameters
        ----------
        n_directions : int, optional (default=None)
            If int, then The top ``n_directions`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

        Returns
        -------
        causal_direction_counts : dict
            List of causal directions sorted by count in descending order.
            The dictionary has the following format::
            {'from': [n_directions], 'to': [n_directions], 'count': [n_directions]}
            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if isinstance(n_directions, (numbers.Integral, np.integer)):
            if not 0 < n_directions:
                raise ValueError(
                    'n_directions must be an integer greater than 0')
        elif n_directions is None:
            pass
        else:
            raise ValueError('n_directions must be an integer greater than 0')

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        # Count causal directions
        directions = []
        for am in np.nan_to_num(self._adjacency_matrices):
            direction = np.array(np.where(np.abs(am) > min_causal_effect))
            if split_by_causal_effect_sign:
                signs = np.array([np.sign(am[i][j])
                                  for i, j in direction.T]).astype('int64').T
                direction = np.vstack([direction, signs])
            directions.append(direction.T)
        directions = np.concatenate(directions)

        if len(directions) == 0:
            cdc = {'from': [], 'to': [], 'count': []}
            if split_by_causal_effect_sign:
                cdc['sign'] = []
            return cdc

        directions, counts = np.unique(directions, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = sort_order[:n_directions] if n_directions is not None else sort_order
        counts = counts[sort_order]
        directions = directions[sort_order]

        cdc = {
            'from': directions[:, 1].tolist(),
            'to': directions[:, 0].tolist(),
            'count': counts.tolist()
        }
        if split_by_causal_effect_sign:
            cdc['sign'] = directions[:, 2].tolist()

        return cdc

    def get_directed_acyclic_graph_counts(self, n_dags=None, min_causal_effect=None, split_by_causal_effect_sign=False):
        """
        Get DAGs count as a result of bootstrapping.

        Parameters
        ----------
        n_dags : int, optional (default=None)
            If int, then The top ``n_dags`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

        Returns
        -------
        directed_acyclic_graph_counts : dict
            List of directed acyclic graphs sorted by count in descending order.
            The dictionary has the following format::
            {'dag': [n_dags], 'count': [n_dags]}.
            where ``n_dags`` is the number of directed acyclic graphs.
        """
        # Check parameters
        if isinstance(n_dags, (numbers.Integral, np.integer)):
            if not 0 < n_dags:
                raise ValueError('n_dags must be an integer greater than 0')
        elif n_dags is None:
            pass
        else:
            raise ValueError('n_dags must be an integer greater than 0')

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        # Count directed acyclic graphs
        dags = []
        for am in np.nan_to_num(self._adjacency_matrices):
            dag = np.abs(am) > min_causal_effect
            if split_by_causal_effect_sign:
                direction = np.array(np.where(dag))
                signs = np.zeros_like(dag).astype('int64')
                for i, j in direction.T:
                    signs[i][j] = np.sign(am[i][j]).astype('int64')
                dag = signs
            dags.append(dag)

        dags, counts = np.unique(dags, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = sort_order[:n_dags] if n_dags is not None else sort_order
        counts = counts[sort_order]
        dags = dags[sort_order]

        if split_by_causal_effect_sign:
            dags = [{
                'from': np.where(dag)[1].tolist(),
                'to': np.where(dag)[0].tolist(),
                'sign': [dag[i][j] for i, j in np.array(np.where(dag)).T]} for dag in dags]
        else:
            dags = [{
                'from': np.where(dag)[1].tolist(),
                'to': np.where(dag)[0].tolist()} for dag in dags]

        return {
            'dag': dags,
            'count': counts.tolist()
        }

    def get_probabilities(self, min_causal_effect=None):
        """
        Get bootstrap probability.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        probabilities : array-like
            List of bootstrap probability matrix.
        """
        # check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        adjacency_matrices = np.nan_to_num(self._adjacency_matrices)
        shape = adjacency_matrices[0].shape
        bp = np.zeros(shape)
        for B in adjacency_matrices:
            bp += np.where(np.abs(B) > min_causal_effect, 1, 0)
        bp = bp/len(adjacency_matrices)

        if int(shape[1]/shape[0]) == 1:
            return bp
        else:
            return np.hsplit(bp, int(shape[1]/shape[0]))

    def get_causal_effects(self, min_causal_effect=None):
        """
        Get total effects list.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        causal_effects : dict
            List of bootstrap causal effect sorted by probability in descending order.
            The dictionary has the following format::
            {'from': [n_directions], 'to': [n_directions], 'effect': [n_directions], 'probability': [n_directions]}
            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        # Calculate probability
        probs = np.sum(np.where(np.abs(self._total_effects) >
                                min_causal_effect, 1, 0), axis=0, keepdims=True)[0]
        probs = probs/len(self._total_effects)

        # Causal directions
        dirs = np.array(np.where(np.abs(probs) > 0))
        probs = probs[dirs[0], dirs[1]]

        # Calculate median effect without zero
        effects = np.zeros(dirs.shape[1])
        for i, (to, from_) in enumerate(dirs.T):
            idx = np.where(np.abs(self._total_effects[:, to, from_]) > 0)
            effects[i] = np.median(self._total_effects[:, to, from_][idx])

        # Sort by probability
        order = np.argsort(-probs)
        dirs = dirs.T[order]
        effects = effects[order]
        probs = probs[order]

        ce = {
            'from': dirs[:, 1].tolist(),
            'to': dirs[:, 0].tolist(),
            'effect': effects.tolist(),
            'probability': probs.tolist()
        }

        return ce
