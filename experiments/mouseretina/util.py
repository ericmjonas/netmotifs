from matplotlib import pylab

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


import irm
import irm.data



def plot_latent(latent, dist_matrix, 
                latent_filename, 
                ground_truth_assign = None, 
                truth_comparison_filename = None):
    """ just getting this code out of pipeline"""

    a = np.array(latent['domains']['d1']['assignment'])

    f = pylab.figure(figsize= (24, 26))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,12])
    
    
    ax = pylab.subplot(gs[1])
    ax_types = pylab.subplot(gs[0])
    
    ai = irm.plot.plot_t1t1_latent(ax, dist_matrix['link'], a)


    # gross_types = np.zeros_like(ground_truth_assign)
    # gross_types[:12] = 0
    # gross_types[12:57] = 1
    # gross_types[58:] = 2 

    # cluster_types = irm.util.compute_purity(a, gross_types)
    # for k, v in cluster_types.iteritems():
    #     print k, ":",  v

    if ground_truth_assign != None:
        for i in  np.argwhere(np.diff(a[ai]) != 0):
            ax_types.axvline(i, c='b', alpha=0.7, linewidth=1.0)

        ax_types.scatter(np.arange(len(ground_truth_assign)), 
                         ground_truth_assign[ai], edgecolor='none', c='k', 
                         s=2)

        ax_types.set_xlim(0, len(ground_truth_assign))
        ax_types.set_ylim(0, 80)
        ax_types.set_xticks([])
        ax_types.set_yticks([])

    f.tight_layout()
    pp = PdfPages(latent_filename)
    f.savefig(pp, format='pdf')


    f2 =  pylab.figure(figsize= (12, 12))
    irm.plot.plot_t1t1_params(f2, dist_matrix, a, 
                              latent['relations']['R1']['ss'], 
                              MAX_DIST=200)
    f2.savefig(pp, format='pdf')
    pp.close()

    if ground_truth_assign != None:

        f = pylab.figure()
        ax_types = pylab.subplot(1, 1, 1)
        irm.plot.plot_purity_ratios(ax_types, a, ground_truth_assign)


        f.savefig(truth_comparison_filename)
