import logging
import numpy as np
import networkx as nx


class SyntheticDataset(object):
    """
    Referred from:
    - - https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, graph_type, degree, sem_type, noise_scale=1.0,
                 dataset_type='nonlinear_1', x_dim=1):
        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.sem_type = sem_type
        self.noise_scale = noise_scale
        self.dataset_type = dataset_type
        self.x_dim = x_dim
        self.w_range = (0.5, 2.0)

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):
        self.W = SyntheticDataset.simulate_random_dag(self.d, self.degree,
                                                      self.graph_type, self.w_range)

        self.X = SyntheticDataset.simulate_sem(self.W, self.n, self.sem_type, self.noise_scale,
                                               self.dataset_type, self.x_dim)

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, w_range):
        """Simulate random DAG with some expected degree.

        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)

        Returns:
            W: weighted DAG
        """
        if graph_type == 'erdos-renyi':
            prob = float(degree) / (d - 1)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == 'full':  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError('unknown graph type')
        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B_perm != 0).astype(float) * U

        return W

    @staticmethod
    def simulate_sem(W, n, sem_type, noise_scale=1.0, dataset_type='nonlinear_1', x_dim=1):
        """Simulate samples from SEM with specified type of noise.

        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM

        Returns:
            X: [n,d] sample matrix
        """
        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d, x_dim])
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            if dataset_type == 'nonlinear_1':
                eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
            elif dataset_type == 'nonlinear_2':
                eta = (X[:, parents, 0] + 0.5).dot(W[parents, j])
            elif dataset_type == 'nonlinear_3':    # Combined version of nonlinear_1 and nonlinear_2
                eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j]) + 0.5
            else:
                raise ValueError('Unknown linear data type')

            if sem_type == 'linear-gauss':
                if dataset_type == 'nonlinear_1':
                    X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
                elif dataset_type in ('nonlinear_2', 'nonlinear_3'):
                    X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n)
            else:
                raise NotImplementedError

        if x_dim > 1 :
            for i in range(x_dim - 1):
                X[:, :, i + 1] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] \
                                + np.random.normal(scale=noise_scale, size=1) \
                                + np.random.normal(scale=noise_scale, size=(n, d))

            X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] \
                            + np.random.normal(scale=noise_scale, size=1) \
                            + np.random.normal(scale=noise_scale, size=(n, d))
        return X


if __name__ == '__main__':
    n, d = 3000, 20
    graph_type, degree, sem_type = 'erdos-renyi', 3, 'linear-gauss'
    noise_scale = 1.0

    dataset = SyntheticDataset(n, d, graph_type, degree, sem_type, noise_scale,
                               dataset_type='nonlinear_1', x_dim=1)

    print('dataset.X.shape: {}'.format(dataset.X.shape))
    print('dataset.W.shape: {}'.format(dataset.W.shape))
