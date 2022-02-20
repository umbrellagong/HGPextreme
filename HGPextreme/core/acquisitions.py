import numpy as np
import scipy.stats as stats


class Acq_Heter(object):
    '''Acquisition for exceeding probability of a stochastic function.   
    '''
    def __init__(self, threshold, input):
        self.threshold = threshold
        self.input = input

    def update_prior_search(self, vhgpr):
        self.vhgpr = vhgpr

    def evaluate(self, position):
        
        mu_f, var_f, mu_g, var_g = self.vhgpr.predict(np.atleast_2d(position))
        
        quads = 1 - stats.norm.cdf(self.threshold, 
                                   [mu_f + np.sqrt(2 * var_f), 
                                    mu_f - np.sqrt(2 * var_f),
                                    mu_f,
                                    mu_f],
                                   [np.sqrt(np.exp(mu_g)),
                                    np.sqrt(np.exp(mu_g)),
                                    np.sqrt(np.exp(mu_g + np.sqrt(2 * var_g))),
                                    np.sqrt(np.exp(mu_g - np.sqrt(2 * var_g)))])
        quads = quads.reshape(-1)
        value = self.input.pdf(position) * np.std(quads)
        
        return - value