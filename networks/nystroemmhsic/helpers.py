import numpy as np
import pandas as pd
import scipy
import time
from networks.nystroemmhsic import data
import networks.nystroemmhsic.estimators, networks.nystroemmhsic.kernels

def gen_data_unif(n, rng, d=1):
#    X = rng.multivariate_normal(mean=np.zeros((d)),cov=np.identity(d), size=n)
    X = rng.uniform(size=(n,d))
    Y = (X[:,0] + rng.normal(size=n)).reshape(-1,1)
    return X, Y

def gen_data_normal(n, rng, d=1):
    X = rng.multivariate_normal(mean=np.zeros((d)),cov=np.identity(d), size=n)  
    Y = (X[:,0] + rng.normal(size=n)).reshape(-1,1)
    return X, Y

def gen_data_weak(n, rng, d=1):
    X = rng.uniform(size=(n,d))
    Y = (rng.normal(size=n)).reshape(-1,1)
    Y[:int(n/4),0] += X[:int(n/4),0]
    return X, Y

def gen_data(n, rng, d=1):
    return gen_data_normal(n=n, rng=rng, d=1)

def standardize(X):
    mx = np.mean(X, 0)
    stdx = np.std(X, axis=0)
    # Assume standard deviations are not 0
    Zx = (X-mx)/stdx
    Zx = np.nan_to_num(Zx)
    assert np.all(np.isfinite(Zx))
    return Zx


def data_MSD(n, rng):
    global msd_df
    
    sample_idx = rng.choice(np.arange(len(msd_df)), size=n, replace=False)
    X = msd_df.drop("Y", axis=1).loc[sample_idx].values
    Y = msd_df["Y"][sample_idx].values.reshape(-1,1)
    
    return standardize(X),standardize(Y)

class Testbench:
    def __init__(self, seed, data_func, null_runs=250, power_runs=100,alpha=0.05):
        self.seed = seed
 
        self.rng = np.random.default_rng(seed)
        self.data_func = data_func
        self.null_runs = null_runs
        self.power_runs = power_runs
        self.alpha=alpha

    def get_bootstrap_indices(self, dataset):
        for _ in range(self.null_runs):
            yield (self.rng.choice(np.arange(len(dataset)), size=len(dataset), replace=True),)
            
    def get_power_runtime(self, estimator, n):
        
        
        start = time.time()
        power = []
        for _ in range(self.power_runs):
            X,Y=self.data_func(n,self.rng)
            
            null_samples = []
            for _ in range(self.null_runs):
                sample_idx = self.rng.choice(np.arange(n), size=n, replace=True)
                null_samples += [estimator(X,Y[sample_idx])]
            
            power += [estimator(X,Y) > np.percentile(null_samples,100-self.alpha*100)]
        runtime = (time.time() - start) / self.power_runs
        
        #power = np.mean(stat_vals > np.percentile(null_samples,100-self.alpha*100))
        return (np.mean(power), runtime)


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

def get_bw(g):
    return np.sqrt(1/(2*g))

import time 

class ContextTimer(object):
    """
    A class used to time an executation of a code snippet. 
    Use it with with .... as ...
    For example, 
        with ContextTimer() as t:
            # do something 
        time_spent = t.secs
    From https://www.huyng.com/posts/python-performance-analysis
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start 
        if self.verbose:
            print('elapsed time: %f ms' % (self.secs*1000))
