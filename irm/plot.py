from matplotlib import pylab
import numpy as np

import scipy.cluster.hierarchy as hier


def plot_zmatrix(ax, zmatrix):
    lm = hier.linkage(zmatrix)
    ord = np.array(hier.leaves_list(lm))
    
    ax.imshow((zmatrix[ord])[:, ord], interpolation='nearest', 
              cmap=pylab.cm.Greys)
    
def plot_purity(ax, true_assignvect, sorted_assign_matrix):
    """
    Plots are best when assign matrix is sorted such that first
    row is most pure, which log score is a good proxy for
    """
    ###
    tv = true_assignvect.argsort()
    tv_i = true_assignvect[tv]
    vals = [tv_i]
    # get the chain order 
    #chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    for a in sorted_assign_matrix:
        a_s = a.argsort(kind='heapsort')
        vals.append(true_assignvect[a_s])
    vals_img = np.vstack(vals)
    ax.imshow(vals_img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(30)
    
