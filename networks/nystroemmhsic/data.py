import numpy as np

def gen_normal_data(n, cov, d=2, seed=1234):
    cov_m = np.ones((d,d))*cov
    np.fill_diagonal(cov_m, 1)

    dat = np.random.default_rng(seed).multivariate_normal(
        mean=[0] * d, cov=cov_m, size=n
    )
    return tuple([dat[:, i].reshape(-1, 1) for i in range(d)])

def dep(n, d=2, seed=1234):
    return gen_normal_data(n=n, cov=1, d=d, seed=seed)
def indep(n, d=2, seed=1234):
    return gen_normal_data(n=n, cov=0, d=d, seed=seed)

def indep_uniform(n, d=2, seed=1234):
    dat = np.random.default_rng(seed=seed).uniform(size=(n,d))
    return tuple([dat[:, i].reshape(-1, 1) for i in range(d)])

    
