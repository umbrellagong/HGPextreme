import sys
sys.path.append("../../")
import warnings
import os

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel as C
from joblib import Parallel, delayed
from core import VHGPR, GaussianInputs, SeqDesign, compute_lh_results
from fourbranches import S, f, r


def main():
    
    threshold = 5
    n_init = 60
    n_step = 3
    n_restart = 6
    opt_grid = 50

    n_trails = 100
    n_total = 150
    n_proc = 25

    # def input
    dim = 2
    mean = 0 * np.ones(dim)
    cov = 1 * np.eye(dim)
    domain = np.array([[-5, 5]]*dim)
    inputs = GaussianInputs(mean, cov, domain, dim)
    np.random.seed(0)
    samples = inputs.sampling(int(2 * 1e5), lh=False)

    # SGPR kernel
    kernel = (C(10, (1e-1, 1e2)) * RBF(5, (1e-1, 1e2)) + 
             WhiteKernel(20, (1e-1, 1e3)))
    # VHGPR kernel
    kernel_f = C(10, (1e-1, 1e2)) * RBF(5, (1e-1, 1e2)) 
    kernel_g = C(2, (1e-1, 1e1)) * RBF(2, (1e-1, 1e1))

    def wrapper_seq_vhgpr(trail, verbose=True):
        warnings.filterwarnings("ignore")
        vhgpr = VHGPR(kernel_f, kernel_g, n_restarts_optimizer=n_restart)
        np.random.seed(trail)
        opt = SeqDesign(S, inputs)
        opt.init_sampling(n_init)
        result = opt.seq_sampling(threshold, n_total, vhgpr, samples, 
                                  n_step, opt_grid=opt_grid, verbose=True)
        if verbose:
            with open('progress', 'a') as f:
                f.write(str(trail) + '\n')
        return result

    def wrapper_lh_vhgpr(trail):
        warnings.filterwarnings("ignore")
        vhgpr = VHGPR(kernel_f, kernel_g, n_restarts_optimizer=n_restart)
        np.random.seed(trail)
        result = compute_lh_results(S, inputs, threshold, vhgpr, samples, 
                                    n_init, n_total, n_step, False)
        return result 

    def wrapper_lh_sgpr(trail):
        warnings.filterwarnings("ignore")
        sgpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=n_restart)
        np.random.seed(trail)
        result = compute_lh_results(S, inputs, threshold, sgpr, samples, 
                                    n_init, n_total, n_step, True)
        return result
    
    def wrapper_true_sgpr_asy(n_very_large=2000):
        warnings.filterwarnings("ignore")
        p_true = (sum(1 - norm.cdf(threshold, f(samples), r(samples))) 
              / samples.shape[0]).item()
        sgpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=n_restart)
        np.random.seed(0)
        p_sgpr_asy = compute_lh_results(S, inputs, threshold, sgpr, samples, 
                                n_very_large, n_very_large, n_step, True)[1][0]
        return p_true, p_sgpr_asy

    os.makedirs('data', exist_ok=True)
    # compute seq_vhgpr, lh_vhgpr, lh_sgpr
    results_seq_vhgpr = Parallel(n_jobs=n_proc)(delayed(wrapper_seq_vhgpr)(j)
                                     for j in range(n_trails))
    np.save('data/seq_vhgpr', np.array(results_seq_vhgpr, dtype=object))

    results_lh_vhgpr = Parallel(n_jobs=n_proc)(delayed(wrapper_lh_vhgpr)(j)
                                     for j in range(n_trails))
    np.save('data/lh_vhgpr', np.array(results_lh_vhgpr, dtype=object))

    results_lh_sgpr = Parallel(n_jobs=n_proc)(delayed(wrapper_lh_sgpr)(j)
                                     for j in range(n_trails))
    np.save('data/lh_sgpr', np.array(results_lh_sgpr, dtype=object))    
    # compute p_true and p_sgpr_asy
    
    np.save('data/const',  wrapper_true_sgpr_asy())

if __name__ == "__main__":
    main()