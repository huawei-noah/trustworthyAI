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

from warnings import warn
import math
import numpy as np
import pandas as pd
from scipy import stats


class CITest(object):
    """
    Class of conditional independence test that contains multiple method

    """

    @staticmethod
    def gauss(data, x, y, z):
        """Gauss test for continues data

        Parameters
        ----------
        data : ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set different
            from x and y. This is the separating set that (potentially)
            makes x and y independent.

        Returns
        -------
        _: None
        _: None
        p: float
            the p-value of conditional independence.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.rand(2500, 4)

        >>> p_value = CITest.cressie_read(data, 0, 1, [])
        >>> print(p_value)
        0.011609430716781555

        >>> p_value = CITest.cressie_read(data, 0, 1, [3])
        >>> print(p_value)
        0.01137523908727811

        >>> p_value = CITest.cressie_read(data, 0, 1, [2, 3])
        >>> print(p_value)
        0.011448214156529746
        """

        n = data.shape[0]
        k = len(z)
        if k == 0:
            r = np.corrcoef(data[:, [x, y]].T)[0][1]
        else:
            sub_index = [x, y]
            sub_index.extend(z)
            sub_corr = np.corrcoef(data[:, sub_index].T)
            # inverse matrix
            try:
                PM = np.linalg.inv(sub_corr)
            except np.linalg.LinAlgError:
                PM = np.linalg.pinv(sub_corr)
            r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
        cut_at = 0.99999
        r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

        # Fisher’s z-transform
        res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
        p_value = 2 * (1 - stats.norm.cdf(abs(res)))

        return None, None, p_value

    @staticmethod
    def g2_test(data, x, y, z):
        """
        G squared test for conditional independence. Also commonly known as G-test,
        likelihood-ratio or maximum likelihood statistical significance test.
        Tests the null hypothesis that x is independent of y given z.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set, different from X and Y.
            This is the separating set that (potentially) makes X and Y independent.

        Returns
        -------
        chi2 : float
            The test statistic.
        dof : int
            Degrees of freedom
        p_value : float
            The p-value of the test

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.randint(0, 5, size=10000).reshape((-1, 4))

        >>> chi2, dof, p_value = CITest.g2_test(data, 0, 1, [])
        >>> print(chi2, dof, p_value)
        20.55310657691933 16 0.19633494733361465

        >>> chi2, dof, p_value = CITest.g2_test(data, 0, 1, [3])
        >>> print(chi2, dof, p_value)
        90.54473365450676 80 0.1971708971451276

        >>> chi2, dof, p_value = CITest.g2_test(data, 0, 1, [2, 3])
        >>> print(chi2, dof, p_value)
        429.0926603059854 400 0.15195497920948475
        """

        return power_divergence(data, x, y, z, lambda_='log-likelihood')

    @staticmethod
    def chi2_test(data, x, y, z):
        """
        Chi-square conditional independence test.

        Tests the null hypothesis that x is independent from y given z.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set, different
            from x and y. This is the separating set that (potentially)
            makes x and y independent.

        Returns
        -------
        chi2 : float
            The test statistic.
        dof : int
            Degrees of freedom
        p_value : float
            The p-value of the test

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.randint(0, 5, size=100).reshape((-1, 4))

        >>> chi2, dof, p_value = CITest.chi2_test(data, 0, 1, [])
        >>> print(chi2, dof, p_value)
        20.542792795683862 16 0.19676171971325737

        >>> chi2, dof, p_value = CITest.chi2_test(data, 0, 1, [3])
        >>> print(chi2, dof, p_value)
        90.66096270618675 80 0.19483257969931803

        >>> chi2, dof, p_value = CITest.chi2_test(data, 0, 1, [2, 3])
        >>> print(chi2, dof, p_value)
        401.830906690841 400 0.46485969015873324
        """

        return power_divergence(data, x, y, z, lambda_='pearson')

    @staticmethod
    def freeman_tukey(data, x, y, z):
        """
        Freeman Tuckey test for conditional independence [1].

        Tests the null hypothesis that x is independent of y given z.

        References
        ----------
        [1] Read, Campbell B. "Freeman—Tukey chi-squared goodness-of-fit
        statistics." Statistics & probability letters 18.4 (1993): 271-278.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set, different
            from x and y. This is the separating set that (potentially)
            makes x and y independent.

        Returns
        -------
        chi2 : float
            The test statistic.
        dof : int
            Degrees of freedom
        p_value : float
            The p-value of the test

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.randint(0, 5, size=10000).reshape((-1, 4))

        >>> chi2, dof, p_value = CITest.freeman_tukey(data, 0, 1, [])
        >>> print(chi2, dof, p_value)
        20.586757281527213 16 0.19494739343907877

        >>> chi2, dof, p_value = CITest.freeman_tukey(data, 0, 1, [3])
        >>> print(chi2, dof, p_value)
        91.06391187965758 80 0.18687227769183953

        >>> chi2, dof, p_value = CITest.freeman_tukey(data, 0, 1, [2, 3])
        >>> print(chi2, dof, p_value)
        nan 400 nan
        """

        return power_divergence(data, x, y, z, lambda_='freeman-tukey')

    @staticmethod
    def modify_log_likelihood(data, x, y, z):
        """
        Modified log likelihood ratio test for conditional independence.

        Tests the null hypothesis that x is independent of y given z.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set, different
            from x and y. This is the separating set that (potentially)
            makes x and y independent.

        Returns
        -------
        chi2 : float
            The test statistic.
        dof : int
            Degrees of freedom
        p_value : float
            The p-value of the test

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.randint(0, 5, size=10000).reshape((-1, 4))

        >>> chi2, dof, p_value = CITest.modify_log_likelihood(data, 0, 1, [])
        >>> print(chi2, dof, p_value)
        20.639717717727184 16 0.19277870421685392

        >>> chi2, dof, p_value = CITest.modify_log_likelihood(data, 0, 1, [3])
        >>> print(chi2, dof, p_value)
        91.97967547179121 80 0.16962335307180806

        >>> chi2, dof, p_value = CITest.modify_log_likelihood(data, 0, 1, [2, 3])
        >>> print(chi2, dof, p_value)
        inf 400 0.0
        """

        return power_divergence(data, x, y, z, lambda_='mod-log-likelihood')

    @staticmethod
    def neyman(data, x, y, z):
        """
        Neyman's test for conditional independence[1].

        Tests the null hypothesis that x is independent of y given z.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Neyman%E2%80%93Pearson_lemma

        Parameters
        ----------
        data : numpy.ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set, different
            from x and y. This is the separating set that (potentially)
            makes x and y independent.

        Returns
        -------
        chi2 : float
            The test statistic.
        dof : int
            Degrees of freedom
        p_value : float
            The p-value of the test

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.randint(0, 5, size=10000).reshape((-1, 4))

        >>> chi2, dof, p_value = CITest.neyman(data, 0, 1, [])
        >>> print(chi2, dof, p_value)
        20.804888528281907 16 0.1861329703686255

        >>> chi2, dof, p_value = CITest.neyman(data, 0, 1, [3])
        >>> print(chi2, dof, p_value)
        95.07200788651971 80 0.11980672825724373

        >>> chi2, dof, p_value = CITest.neyman(data, 0, 1, [2, 3])
        >>> print(chi2, dof, p_value)
        nan 400 nan
        """

        return power_divergence(data, x, y, z, lambda_='neyman')

    @staticmethod
    def cressie_read(data, x, y, z):
        """
        Cressie Read statistic for conditional independence[1].

        Tests the null hypothesis that x is independent of y given z.

        References
        ----------
        [1] Cressie, Noel, and Timothy RC Read.
        "Multinomial goodness‐of‐fit tests." Journal of the Royal Statistical
        Society: Series B (Methodological) 46.3 (1984): 440-464.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set, different
            from x and y. This is the separating set that (potentially)
            makes x and y independent.

        Returns
        -------
        chi2 : float
            The test statistic.
        dof : int
            Degrees of freedom
        p_value : float
            The p-value of the test

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.randint(0, 5, size=10000).reshape((-1, 4))

        >>> chi2, dof, p_value = CITest.cressie_read(data, 0, 1, [])
        >>> print(chi2, dof, p_value)
        20.537851851639562 16 0.19696641879639076

        >>> chi2, dof, p_value = CITest.cressie_read(data, 0, 1, [3])
        >>> print(chi2, dof, p_value)
        90.45257795422611 80 0.19903833818274186

        >>> chi2, dof, p_value = CITest.cressie_read(data, 0, 1, [2, 3])
        >>> print(chi2, dof, p_value)
        404.24753197461905 400 0.43124831946260705
        """

        return power_divergence(data, x, y, z, lambda_='cressie-read')


def power_divergence(data, x, y, z, lambda_=None):
    """
    This function tests the null hypothesis that the categorical data.

    The null hypothesis for the test is x is independent of y given z.
    A lot of the frequency comparison based statistics
    (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.


    Parameters
    ----------
    data : numpy.ndarray
        The dataset on which to test the independence condition.
    x : int
        A variable in data set
    y : int
        A variable in data set
    z : List, default []
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic [2]_.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family [3]_ to be used instead.
        For convenience, `lambda_` may be assigned one of the following
        strings, in which case the corresponding numerical value is used::

            String              Value   Description
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test [3]_.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   The power recommended in [5]_.

    Returns
    -------
    chi2 : float
        The test statistic.
    dof : int
        Degrees of freedom
    p_value : float
            The p-value of the test

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171015035606/http://faculty.vassar.edu/lowry/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    .. [3] "G-test", https://en.wikipedia.org/wiki/G-test
    .. [4] Sokal, R. R. and Rohlf, F. J. "Biometry: the principles and
           practice of statistics in biological research", New York: Freeman
           (1981)
    .. [5] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.

    See Also
    --------
    scipy.stats.power_divergence
    scipy.stats.chi2_contingency

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(23)
    >>> data = np.random.randint(0, 5, size=100).reshape((-1, 4))
    >>> data = np.concatenate([data, data.sum(axis=1).reshape(-1, 1)], axis=1)

    >>> chi2, dof, p_value = power_divergence(data, 0, 1, [])
    >>> print(chi2, dof, p_value)
    >>> 16.005291005291006 16 0.45259159404543464

    >>> chi2, dof, p_value = power_divergence(data, 0, 1, [3])
    >>> print(chi2, dof, p_value)
    >>> 25.333333333333336 25 0.4438225249645223

    >>> chi2, dof, p_value = power_divergence(data, 0, 1, [3, 4])
    >>> print(chi2, dof, p_value)
    >>> 0.0 5 1.0
    """

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if len(z) == 0:
        group_x_y = data.groupby([x, y]).size()
        x_y = group_x_y.unstack(y, fill_value=0)
        chi2, p_value, dof, exp = stats.chi2_contingency(x_y, lambda_=lambda_)
    else:
        chi2 = 0
        dof = 0
        for z_state, df in data.groupby(z):
            try:
                group_x_y = df.groupby([x, y]).size()
                x_y = group_x_y.unstack(y, fill_value=0)
                c, _, d, _ = stats.chi2_contingency(x_y, lambda_=lambda_)
                chi2 += c
                dof += d
            except ValueError:
                warn(f"Skipping the test {x}\u27C2{y}|{z}. Not enough samples.")
        if dof == 0:
            p_value = 1.0
        else:
            p_value = stats.chi2.sf(chi2, df=dof)
                
    return chi2, dof, p_value


def _rbf_dot(x):

    n = x.shape[0]

    G = np.sum(x * x, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    H = Q + G.T - 2 * np.dot(x, x.T)

    dists = Q + G.T - 2 * np.dot(x, x.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)
    deg = np.sqrt(0.5 * np.median(dists[dists > 0]))

    H = np.exp(-H / 2 / (deg ** 2))

    return H


def hsic_test(x, y, alpha=0.05, normalize=True):
    """Hilbert-Schmidt independence criterion

    HSIC with a Gaussian kernel for the independence test,
    where we used the gamma distribution as an approximation for the
    distribution of the HSIC.

    References
    ----------
    https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf

    Parameters
    ----------
    x: numpy array
        Data of the first variable. (n, dim_x) numpy array.
    y: numpy array
        Data of the second variable. (n, dim_y) numpy array.
    alpha : float, default 0.05
        significance level
    normalize: bool, default True
        whether use data normalization

    Returns
    -------
    out: int, 0 or 1
        If 0, x and y are independent.
        If 1, x and y are not independent.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = np.random.rand(500, 2)
    >>> print(hsic_test(x[:, [0]], x[:, [1]]))
    1

    >>> np.random.seed(12)
    >>> x = np.random.rand(500, 2)
    >>> print(hsic_test(x[:, [0]], x[:, [1]]))
    0
    """

    if normalize:
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

    n = x.shape[0]

    H = np.identity(n) - np.ones((n, n), dtype=float) / n
    K = _rbf_dot(x)
    L = _rbf_dot(y)
    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6) ** 2
    varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)
    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    bone = np.ones((n, 1), dtype=float)
    muX = np.dot(np.dot(bone.T, K), bone) / n / (n - 1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n - 1)
    mHSIC = (1 + muX * muY - muX - muY) / n
    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * n / mHSIC

    thresh = stats.gamma.ppf(1 - alpha, al, scale=bet)[0][0]

    if testStat < thresh:
        return 1
    else:
        return 0
