import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class Kernel:
    pass


class TranslationInvariant:
    def Z(self, X, num_rff, rng):
        pass



class RBF(Kernel, TranslationInvariant):
    def __init__(self, gamma):
        self.gamma = gamma
        self.w = []

    def k(self, X, Y=None):
        return rbf_kernel(X=X, Y=Y, gamma=self.gamma)

    def Z(self, X, num_rff, rng):
        w = rng.normal(scale=self.sigma_from_gamma(self.gamma),size=(num_rff,X.shape[1]))
        template = X @ w.T
        return 1/np.sqrt(num_rff) * np.concatenate((np.cos(template.T),np.sin(template.T))).T

    @staticmethod
    def estimate_gamma(X, n_samples=1000, seed=1234):
        """Estimate the gamma parameter based on the median heuristic for sigma**2.
        """
        rng = np.random.default_rng(seed)
        distances = []
        for i in range(n_samples):
            distances += [np.linalg.norm(rng.choice(X, size=2)) ** 2]
        sigma = np.median(distances)
        return 1 / np.sqrt(2 * sigma)

    @staticmethod
    def sigma_from_gamma(gamma):
        return np.sqrt(2*gamma)
