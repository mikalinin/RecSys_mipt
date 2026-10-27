import numpy as np

class EASE:
    def __init__(self, regularization=300.):
        self.reg = regularization

    def fit(self, positive):
        self.X = positive.tocsr().astype(float)
        gram = (self.X.T @ self.X).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        inverse = np.linalg.solve(gram, np.eye(gram.shape[0]))
        self.P = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.P, 0.)
        return self

    def score(self, users):
        return self.X[users] @ self.P
