import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def plot_results(title, ylim=None, xlim=None, whether_save=True):
    
    plt.figure(figsize=(6,4))
    
    for method in ('lh_sgpr', 'lh_vhgpr', 'seq_vhgpr'):
        results = np.load('data/' + method + '.npy', allow_pickle=True)
        x = results[0][0]
        y = [i[1] for i in results]
        plt.plot(x, np.mean(y, axis=0))
        plt.fill_between(x, np.mean(y, axis =0), 
                         np.mean(y, axis=0) + np.std(y, axis=0), 
                         alpha = 0.2)
    p_true, p_sgp_asy = np.load('data/const.npy', allow_pickle=True)
    plt.plot(x, np.ones(len(x)) * p_true *(1+0.1),"k--")
    plt.plot(x, np.ones(len(x)) * p_true *(1-0.1),"k--")
    plt.plot(x, np.ones(len(x)) * p_sgp_asy, ':', color = 'purple')
    plt.xlabel('Number of Samples')
    plt.ylabel('$P_e$')
    plt.title(title)
    plt.ylim(ylim)
    plt.xlim(xlim)
    if whether_save:
        plt.savefig('result.pdf', bbox_inches = "tight")
    plt.show()