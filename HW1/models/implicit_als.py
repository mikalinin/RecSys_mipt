import numpy as np
from scipy import sparse

class ImplicitALS:
    def __init__(self, factors=16, iterations=4, alpha=20., reg=1., seed=2026):
        self.factors, self.iterations, self.alpha, self.reg, self.seed = factors, iterations, alpha, reg, seed

    def _solve(self, target, confidence, fixed):
        gram = fixed.T @ fixed + self.reg * np.eye(self.factors)
        out = np.zeros((target.shape[0], self.factors))
        active = np.flatnonzero(np.diff(target.indptr))
        for start in range(0, len(active), 512):
            rows = active[start:start + 512]
            a = np.broadcast_to(gram, (len(rows), *gram.shape)).copy()
            b = np.zeros((len(rows), self.factors))
            for j, row in enumerate(rows):
                s, e = target.indptr[row:row + 2]
                y = fixed[target.indices[s:e]]
                c, p = confidence.data[s:e], target.data[s:e]
                a[j] += y.T @ ((c - 1)[:, None] * y)
                b[j] = y.T @ (c * p)
            out[rows] = np.linalg.solve(a, b[..., None])[..., 0]
        return out

    def fit(self, positive, target=None):
        p = positive.tocsr().astype(float) if target is None else target.tocsr().astype(float)
        p.sort_indices()
        c = p.copy()
        c.data[:] = 1 + self.alpha
        rng = np.random.default_rng(self.seed)
        y = rng.normal(0, .01, (p.shape[1], self.factors))
        pt, ct = p.T.tocsr(), c.T.tocsr()
        for _ in range(self.iterations):
            x = self._solve(p, c, y)
            y = self._solve(pt, ct, x)
        self.X, self.Y = x, y
        return self

    def score(self, users):
        return self.X[users] @ self.Y.T
