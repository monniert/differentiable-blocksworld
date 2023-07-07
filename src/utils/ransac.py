import torch

from . import use_seed
from .logger import print_log


class LSLinearRegressor:
    def __init__(self, params=None, index=None):
        if index is not None:
            self.params = params[index]
        else:
            self.params = params

    def fit(self, X, y):
        X = torch.cat([torch.ones_like(X)[..., 0:1], X], dim=-1)
        Xt = X.transpose(-2, -1)
        self.params = (Xt @ X).inverse() @ Xt @ y
        return self

    def predict(self, X, index=None):
        X = torch.cat([torch.ones_like(X)[..., 0:1], X], dim=-1)
        if index is not None:
            return X @ self.params[index]
        else:
            return X @ self.params

    def clone(self, index=None):
        return self.__class__(params=self.params.clone(), index=index)


class Ransac:
    def __init__(self, n_points=3, n_iter=100, batch_size=10, thresh=0.001, model=None):
        self.n_points = n_points
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.thresh = thresh
        self.model = model or LSLinearRegressor()
        self.best_models = []
        self.best_n = 0

    @use_seed()
    def fit(self, X, y):
        N, P, B = X.size(0), self.n_points, self.batch_size
        n_batch = self.n_iter // B
        for k in range(n_batch):
            idxs = torch.randint(0, N, (B * P,))
            self.model.fit(X[idxs].view(B, P, -1), y[idxs].view(B, P, -1))
            diff = (y[None].expand(B, -1, -1) - self.model.predict(X[None].expand(B, -1, -1))).flatten(1)
            inliers = diff.pow(2) < self.thresh
            n_inliers = inliers.sum(1)

            best_idx = torch.argmax(n_inliers)
            best_n = n_inliers[best_idx].item()
            if best_n > self.best_n:
                print_log(f'Ransac iter={k*self.batch_size}: found a best model with {best_n} inliers')
                self.best_n = best_n
                self.best_models.append(self.model.clone(best_idx))
