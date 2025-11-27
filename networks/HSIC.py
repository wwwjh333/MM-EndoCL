import torch.nn as nn
import torch

class HSIC(nn.Module):
    def __init__(self,kernel):
        super(HSIC, self).__init__()
        self.kernels = kernel
    def forward(self, X, Y):
        kernelX = self.kernels
        kernelY = self.kernels
        n = len(X)
        K = kernelX(X)
        L = kernelY(Y)

        H = torch.eye(n) - (1 / n)
        H = H.to(X.device)

        KH = torch.matmul(K, H)
        LH = torch.matmul(L, H)
        KHLH = torch.matmul(KH, LH)

        return 1 / (n ** 2) * torch.trace(KHLH)

class RBF(nn.Module):
    def __init__(self, gamma=1e-3):
        super(RBF, self).__init__()
        self.gamma = gamma

    def forward(self, X, Y=None):
        return self.rbf_kernel(X=X, Y=Y, gamma=self.gamma)

    def rbf_kernel(self,X, Y=None, gamma=None):
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        if Y is None:
            Y = X
        dist = torch.cdist(X, Y, p=2) ** 2

        K = torch.exp(-gamma * dist)
        return K

def est_sigma(X, max_len=500):
    n = min(len(X), max_len)
    dists = []
    for i in range(n):
        for j in range(i,n):
            dists += [np.linalg.norm(X[i]-X[j],ord=2)**2]
    bw = np.median(dists)
    return np.sqrt(bw*0.5)

def g(bw):
    return 1/(2*bw**2)