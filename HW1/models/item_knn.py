import numpy as np
from scipy import sparse

class ItemKNN:
    def __init__(self, neighbors=40):
        self.neighbors = neighbors

    def fit(self, positive):
        self.X = positive.tocsr().astype(float)
        norms = np.sqrt(np.asarray(self.X.power(2).sum(axis=0)).ravel())
        normalized = self.X @ sparse.diags(1 / np.maximum(norms, 1e-12))
        sim = (normalized.T @ normalized).toarray()
        np.fill_diagonal(sim, 0)
        k = min(self.neighbors, max(0, sim.shape[1] - 1))
        order = np.argsort(-sim, axis=1, kind='stable')
        if k < sim.shape[1]:
            np.put_along_axis(sim, order[:, k:], 0., axis=1)
        self.sim = sparse.csr_matrix(sim)
        return self

    def score(self, users):
        return (self.X[users] @ self.sim).toarray()
