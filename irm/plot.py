import numpy as np

import scipy.cluster.hierarchy as hier
import util


def plot_zmatrix(ax, zmatrix):
    from matplotlib import pylab

    lm = hier.linkage(zmatrix)
    ord = np.array(hier.leaves_list(lm))
    
    ax.imshow((zmatrix[ord])[:, ord], interpolation='nearest', 
              cmap=pylab.cm.Greys)
    return ord
    
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
        #a_s = a.argsort(kind='heapsort')
        # a more concerted effort to mix things up
        out = np.zeros_like(a)
        pos = 0
        for c in np.unique(a):
            eq_c= np.argwhere(a == c).flatten()
            out[pos:pos+len(eq_c)] = np.random.permutation(eq_c)
            pos += len(eq_c)
        
        vals.append(true_assignvect[out])
    vals_img = np.vstack(vals)
    ax.imshow(vals_img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(30)
    

def plot_purity_ratios(ax, clustering, truth):
    """
    For a given assignment vector, plot, for each true cluster types, 
    how many different clusters it was in. 
    
    Sort by true cluster size
    """
    
    true_order, true_sizes, fracs_order = util.compute_purity_ratios(clustering, truth)


    left = np.cumsum(np.hstack([np.array([0]), true_sizes]))[:-1]
    height = [f[0] for f in fracs_order]
    ax.bar(left, height, width= true_sizes)
    
    
