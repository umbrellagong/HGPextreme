import copy
import numpy as np
from scipy import optimize
from .metrics import fail_prob
from .acquisitions import Acq_Heter



class SeqDesign(object):
    def __init__(self, S, inputs):
        self.S = S
        self.inputs = inputs

    def init_sampling(self, n_init): 
        self.DX = self.inputs.sampling(n_init)
        self.DY = self.S(self.DX)
        self.n_init = n_init
        return self
    
    def load(self, X, Y): 
        self.DX = np.copy(X)
        self.DY = np.copy(Y)
        self.n_init = Y.shape[0]
        return self

    def seq_sampling(self, threshold, n_total, model, samples, n_step=3, 
                     perturb_std=0.02, opt_grid=40, weights=None, 
                     verbose=False):

        acq = Acq_Heter(threshold, self.inputs)
        model = copy.deepcopy(model)
        trace_p = []
        trace_n = []

        for i in range(self.n_init, n_total+n_step, n_step):
            model.fit(self.DX, self.DY)
            trace_p.append(fail_prob(model, threshold, samples, False, weights))
            trace_n.append(i)
            if i >= n_total:
                if verbose:
                    print('n: ', trace_n[-1], " p: ", trace_p[-1])
                break
            acq.update_prior_search(model)
            best_pos = optimize.brute(acq.evaluate, self.inputs.domain, 
                                      Ns=opt_grid, full_output=False, 
                                      finish=None)
            if verbose:
                print('n: ', trace_n[-1], " p: ", trace_p[-1], " x: ", best_pos)
            best_pos = np.random.multivariate_normal(np.atleast_1d(best_pos), 
                                        np.eye(self.inputs.dim) * perturb_std, 
                                        n_step)
            self.DX = np.append(self.DX, best_pos, axis=0)
            self.DY = np.append(self.DY, self.S(best_pos))
        
        return trace_n, trace_p, self.DX, self.DY


def compute_lh_results(S, inputs, threshold, model, samples, n_start, n_total, 
                       n_step, whether_sgp, weights=None):

    DX = inputs.sampling(n_total)
    DY = S(DX)
    model = copy.deepcopy(model)
    trace_p = []
    trace_n = []
    for i in range(n_start, n_total + n_step, n_step):
        model.fit(DX[:i], DY[:i])
        trace_p.append(fail_prob(model, threshold, samples, whether_sgp, 
                                 weights))
        trace_n.append(i)

    return trace_n, trace_p, DX, DY
        