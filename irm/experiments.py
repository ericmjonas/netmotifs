import cPickle as pickle
import numpy as np
import copy
import os, glob
import time

import irm
import irm.data
import util


"""
Code to run experiments -- this is shared across a lot of them. This is NOT
production-quality code by any means and is a grab-bag of "crap we use" 

It's only here because it's easy to import. 

"""


def create_init(latent_filename, data_filename, out_filenames, 
                init= None):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)
    """
    irm_latent = pickle.load(open(latent_filename, 'r'))
    irm_data = pickle.load(open(data_filename, 'r'))
    irm_latents = []

    rng = irm.RNG()

    irm_model = irm.irmio.create_model_from_data(irm_data, rng=rng)
    for c, out_f in enumerate(out_filenames):
        print "generating init", out_f
        np.random.seed(c)

        latent = copy.deepcopy(irm_latent)

        if init['type'] == 'fixed':
            group_num = init['group_num']

            a = np.arange(len(latent['domains']['d1']['assignment'])) % group_num
            a = np.random.permutation(a)

        elif init['type'] == 'crp':
            alpha = init['alpha']
            raise NotImplementedError("crp not done")
        elif init['type'] == 'truth':
            a = latent['domains']['d1']['assignment']
            
        else:
            raise NotImplementedError("Unknown init type")
            
        if c > 0: # first one stays the same
            latent['domains']['d1']['assignment'] = a

        # generate new suffstats, recompute suffstats in light of new assignment

        irm.irmio.set_model_latent(irm_model, latent, rng)

        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=10)


        pickle.dump(irm.irmio.get_latent(irm_model), open(out_f, 'w'))


def plot_latent(latent, dist_matrix, 
                latent_filename, 
                ground_truth_assign = None, 
                truth_comparison_filename = None, 
                model='LogisticDistance', PLOT_MAX_DIST=200):
    """ just getting this code out of pipeline"""
    from matplotlib import pylab

    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

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

    # SUCH A HACK SHOULD PASS IN DATA. GOD THIS CODE IS TURNING TO SHIT
    # UNDER TIME PRESSURE

    f2 =  pylab.figure(figsize= (12, 12))
    irm.plot.plot_t1t1_params(f2, dist_matrix, a, 
                              latent['relations']['R1']['ss'], 
                              MAX_DIST=PLOT_MAX_DIST, model=model)
    f2.savefig(pp, format='pdf')


    pp.close()

    if ground_truth_assign != None:

        f = pylab.figure()
        ax_types = pylab.subplot(1, 1, 1)
        irm.plot.plot_purity_ratios(ax_types, a, ground_truth_assign)


        f.savefig(truth_comparison_filename)
