import numpy as np
from scipy.special import expit

class FM:
    def __init__(self, factors=8, epochs=4, lr=.01, reg=1e-5, seed=2026,
                 max_samples_per_epoch=60000, use_metadata=True, batch_size=1024):
        self.factors, self.epochs, self.lr, self.reg, self.seed = factors, epochs, lr, reg, seed
        self.max_samples = max_samples_per_epoch
        self.use_metadata, self.batch_size = use_metadata, batch_size

    def _meta(self, positive, events):
        U, I = positive.shape
        def buckets(values):
            return np.searchsorted(np.quantile(values, [.2,.4,.6,.8]), values, side='right')
        uf, itf = [np.arange(U)], [U + np.arange(I)]
        offset = U + I
        if self.use_metadata:
            for col, n, fields in [('u', U, uf), ('i', I, itf)]:
                grouped = events.groupby(col)
                for values in [grouped.size().reindex(range(n), fill_value=0).to_numpy(),
                               grouped.rating.mean().reindex(range(n), fill_value=3).to_numpy(),
                               grouped.verified_purchase.mean().reindex(range(n), fill_value=0).to_numpy()]:
                    fields.append(offset + buckets(values))
                    offset += 5
        self.uf, self.itf = np.column_stack(uf), np.column_stack(itf)
        return offset

    def fit(self, positive, events):
        rng = np.random.default_rng(self.seed)
        positive = positive.tocsr()
        n = self._meta(positive, events)
        self.w0, self.w = 0., np.zeros(n)
        self.v = np.random.default_rng(self.seed).normal(0, .01, (n, self.factors))
        mw, vw = np.zeros(n), np.zeros(n)
        mv, vv = np.zeros_like(self.v), np.zeros_like(self.v)
        mb = vb = 0.
        pu, pi = positive.nonzero()
        observed = positive.copy()
        observed = observed.tolil()
        for u, i in zip(events.u, events.i):
            observed[u, i] = 1
        observed = observed.tocsr()
        eligible = np.flatnonzero((np.diff(observed.indptr) < positive.shape[1]) &
                                  (np.diff(positive.indptr) > 0))
        step = 0
        for _ in range(self.epochs):
            count = min(len(pu), self.max_samples // 2)
            pick = rng.choice(len(pu), count, replace=False)
            nu = rng.choice(eligible, count)
            ni = rng.integers(positive.shape[1], size=count)
            bad = np.asarray(observed[nu, ni]).ravel() > 0
            while bad.any():
                ni[bad] = rng.integers(positive.shape[1], size=int(bad.sum()))
                bad = np.asarray(observed[nu, ni]).ravel() > 0
            u, i = np.r_[pu[pick], nu], np.r_[pi[pick], ni]
            labels = np.r_[np.ones(count), np.zeros(count)]
            order = rng.permutation(len(u))
            for s in range(0, len(u), self.batch_size):
                batch = order[s:s+self.batch_size]
                ids = np.column_stack([self.uf[u[batch]], self.itf[i[batch]]])
                values = self.v[ids]
                sums = values.sum(axis=1)
                pred = self.w0 + self.w[ids].sum(axis=1) + .5 * (
                    (sums*sums).sum(axis=1) - (values*values).sum(axis=(1,2)))
                error = (expit(pred) - labels[batch]) / len(batch)
                gw, gv = self.reg*self.w, self.reg*self.v
                np.add.at(gw, ids.ravel(), np.repeat(error, ids.shape[1]))
                np.add.at(gv, ids.ravel(), (error[:,None,None]*(sums[:,None,:]-values)).reshape(-1,self.factors))
                step += 1
                for param, grad, m, v in [(self.w,gw,mw,vw),(self.v,gv,mv,vv)]:
                    m *= .9; m += .1*grad
                    v *= .999; v += .001*grad*grad
                    param -= self.lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
                g = error.sum()
                mb, vb = .9*mb+.1*g, .999*vb+.001*g*g
                self.w0 -= self.lr*(mb/(1-.9**step))/(np.sqrt(vb/(1-.999**step))+1e-8)
        return self

    def score(self, users):
        def components(ids):
            v = self.v[ids]
            sums = v.sum(axis=1)
            bias = self.w[ids].sum(axis=1) + .5*((sums*sums).sum(axis=1)-(v*v).sum(axis=(1,2)))
            return sums, bias
        u, ub = components(self.uf[users])
        i, ib = components(self.itf)
        return self.w0 + ub[:,None] + ib[None,:] + u @ i.T
