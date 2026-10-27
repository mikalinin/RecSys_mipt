import numpy as np
from scipy import sparse
from .implicit_als import ImplicitALS

class CustomALS(ImplicitALS):
    def __init__(self, half_life_days=365, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def fit(self, positive, events):
        e = events[events.rating >= 4]
        age_days = (events.timestamp.max() - e.timestamp.to_numpy()) / 86400
        target = ((e.rating.to_numpy() - 3) / 2) * (
            .5 + .5 * np.exp2(-age_days / self.half_life_days))
        p = sparse.csr_matrix((target, (e.u, e.i)), shape=positive.shape)
        return super().fit(positive, target=p)
