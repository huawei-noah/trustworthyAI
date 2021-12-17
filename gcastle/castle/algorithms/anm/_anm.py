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
from sklearn.preprocessing import scale
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import hsic_test


class GPR(object):
    """Estimator based on Gaussian Process Regressor

    Parameters
    ----------
    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. Note that this is
        different from using a `WhiteKernel`. If an array is passed, it must
        have the same number of entries as the data used for fitting and is
        used as datapoint-dependent noise level. Allowing to specify the
        noise level directly as a parameter is mainly for convenience and
        for consistency with Ridge.

    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel ``ConstantKernel(1.0, constant_value_bounds="fixed"
        * RBF(1.0, length_scale_bounds="fixed")`` is used as default. Note that
        the kernel hyperparameters are optimized during fitting unless the
        bounds are marked as "fixed".

    optimizer : "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be minimized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'L-BGFS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    normalize_y : bool, default=False
        Whether the target values y are normalized, the mean and variance of
        the target values are set equal to 0 and 1 respectively. This is
        recommended for cases where zero-mean, unit-variance priors are used.
        Note that, in this implementation, the normalisation is reversed
        before the GP predictions are reported.

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.

    See Also
    --------
    from sklearn.gaussian_process import GaussianProcessRegressor

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(10).reshape((-1, 1))
    >>> y = np.random.rand(10).reshape((-1, 1))
    >>> gpr = GPR(alpha=1e-10)
    >>> y_pred = gpr.estimate(x, y)
    >>> print(y_pred)
    [[0.30898833]
     [0.51335394]
     [0.378371  ]
     [0.47051942]
     [0.51290679]
     [0.29678631]
     [0.77848816]
     [0.47589755]
     [0.21743226]
     [0.35258412]]
    """

    def __init__(self, **kwargs):
        super(GPR, self).__init__()
        self.regressor = GaussianProcessRegressor(**kwargs)

    def estimate(self, x, y):
        """Fit Gaussian process regression model and predict x.

        Parameters
        ----------
        x : array
            Variable seen as cause
        y: array
            Variable seen as effect

        Returns
        -------
        y_predict: array
            regression predict values of x
        """

        self.regressor.fit(x, y)
        y_predict = self.regressor.predict(x)

        return y_predict


class ANMNonlinear(BaseLearner):
    """
    Nonlinear causal discovery with additive noise models

    Use GPML with Gaussian kernel and independent Gaussian noise,
    optimizing the hyper-parameters for each regression individually.
    For the independence test, we implemented the HSIC with a Gaussian kernel,
    where we used the gamma distribution as an approximation for the
    distribution of the HSIC under the null hypothesis of independence
    in order to calculate the p-value of the test result.

    References
    ----------
    Hoyer, Patrik O and Janzing, Dominik and Mooij, Joris M and Peters,
    Jonas and Schölkopf, Bernhard,
    "Nonlinear causal discovery with additive noise models", NIPS 2009

    Parameters
    ----------
    alpha : float, default 0.05
        significance level be used to compute threshold

    Attributes
    ----------
    causal_matrix : array like shape of (n_features, n_features)
        Learned causal structure matrix.

    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import DAG, IIDSimulation
    >>> from castle.algorithms.anm import ANMNonlinear

    >>> weighted_random_dag = DAG.erdos_renyi(n_nodes=6, n_edges=10,
    >>>                                      weight_range=(0.5, 2.0), seed=1)
    >>> dataset = IIDSimulation(W=weighted_random_dag, n=1000,
    >>>                         method='nonlinear', sem_type='gp-add')
    >>> true_dag, X = dataset.B, dataset.X

    >>> anm = ANMNonlinear(alpha=0.05)
    >>> anm.learn(data=X)

    >>> # plot predict_dag and true_dag
    >>> GraphDAG(anm.causal_matrix, true_dag, show=False, save_name='result')

    you can also provide more parameters to use it. like the flowing:
    >>> from sklearn.gaussian_process.kernels import Matern, RBF
    >>> kernel = Matern(nu=1.5)
    >>> # kernel = 1.0 * RBF(1.0)
    >>> anm = ANMNonlinear(alpha=0.05)
    >>> anm.learn(data=X, regressor=GPR(kernel=kernel))
    >>> # plot predict_dag and true_dag
    >>> GraphDAG(anm.causal_matrix, true_dag, show=False, save_name='result')
    """

    def __init__(self, alpha=0.05):
        super(ANMNonlinear, self).__init__()
        self.alpha = alpha

    def learn(self, data, columns=None, regressor=GPR(), test_method=hsic_test, **kwargs):
        """Set up and run the ANM_Nonlinear algorithm.

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        regressor: Class
            Nonlinear regression estimator, if not provided, it is GPR.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        """

        self.regressor = regressor

        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        node_num = data.shape[1]
        self.causal_matrix = Tensor(np.zeros((node_num, node_num)),
                                    index=data.columns,
                                    columns=data.columns)

        for i, j in combinations(range(node_num), 2):
            x = data[:, i].reshape((-1, 1))
            y = data[:, j].reshape((-1, 1))

            flag = test_method(x, y, alpha=self.alpha)
            if flag == 1:
                continue
            # test x-->y
            flag = self.anm_estimate(x, y, regressor=regressor,
                                     test_method=test_method)
            if flag:
                self.causal_matrix[i, j] = 1
            # test y-->x
            flag = self.anm_estimate(y, x, regressor=regressor,
                                     test_method=test_method)
            if flag:
                self.causal_matrix[j, i] = 1

    def anm_estimate(self, x, y, regressor=GPR(), test_method=hsic_test):
        """Compute the fitness score of the ANM model in the x->y direction.

        Parameters
        ----------
        x: array
            Variable seen as cause
        y: array
            Variable seen as effect
        regressor: Class
            Nonlinear regression estimator, if not provided, it is GPR.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`

        Returns
        -------
        out: int, 0 or 1
            If 1, residuals n is independent of x, then accept x --> y
            If 0, residuals n is not independent of x, then reject x --> y

        Examples
        --------
        >>> import numpy as np
        >>> from castle.algorithms.anm import ANMNonlinear
        >>> np.random.seed(1)
        >>> x = np.random.rand(500, 2)
        >>> anm = ANMNonlinear(alpha=0.05)
        >>> print(anm.anm_estimate(x[:, [0]], x[:, [1]]))
        1
        """

        x = scale(x).reshape((-1, 1))
        y = scale(y).reshape((-1, 1))

        y_predict = regressor.estimate(x, y)
        flag = test_method(y - y_predict, x, alpha=self.alpha)

        return flag
