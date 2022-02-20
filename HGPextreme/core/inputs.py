import numpy as np
import scipy.stats as stats
from pyDOE import lhs


class GaussianInputs: 
    'A class for Gaussian inputs'
    def __init__(self, mean, cov, domain, dim):
        self.mean = mean
        self.cov = cov
        self.domain = domain
        self.dim = dim
    def sampling(self, num, lh=True, criterion=None):
        if lh:
            samples = lhs(self.dim, num, criterion=criterion)
            samples = self.rescale_samples(samples, self.domain)
        else: 
            samples = np.random.multivariate_normal(self.mean, self.cov, num)
        return samples
    def pdf(self, x):
        return stats.multivariate_normal(self.mean, self.cov).pdf(x) 

    @staticmethod
    def rescale_samples(x, domain):
        """Rescale samples from [0,1]^d to actual domain."""
        for i in range(x.shape[1]):
            bd = domain[i]
            x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
        return x