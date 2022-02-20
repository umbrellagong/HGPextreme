import numpy as np
import scipy.stats as stats

def fail_prob(model, threshold, samples, whether_sgp, weights=None):
    '''compute exceeding probability
    '''
    n_samples = samples.shape[0]
    n_parts = int(np.floor(n_samples / 1e5))
    samples_list = np.array_split(samples, n_parts)
    if whether_sgp:
        f = np.empty(0)
        for i in range(n_parts):
            f_ = model.predict(samples_list[i])
            f = np.concatenate((f, f_))
        probs = 1 - stats.norm.cdf(threshold, f, 
                                   np.sqrt(np.exp(model.kernel_.theta[2])))
    else:
        f = np.empty(0)
        g = np.empty(0)
        for i in range(n_parts):
            f_, _, g_, _ = model.predict(samples_list[i])
            f = np.concatenate((f, f_))
            g = np.concatenate((g, g_))
        probs = 1 - stats.norm.cdf(threshold, f, np.sqrt(np.exp(g)))

    if weights==None:
        prob_all = np.sum(probs) / samples.shape[0]
    else:
        prob_all = np.sum(probs) * weights / np.sum(weights)

    return prob_all
