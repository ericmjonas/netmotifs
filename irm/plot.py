import numpy as np
import scipy.cluster.hierarchy as hier
import util
import cPickle as pickle

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
    
    
def plot_t1t1_latent(ax, adj_matrix, assign_vect):
    """
    Plot a latent with the assign vect

    returns the sorted order of the assignment vector
    """

    from matplotlib import pylab

    a = util.canonicalize_assignment(assign_vect) # make big clusters first

    ai = np.argsort(a).flatten()
        
    conn = adj_matrix

    s_conn =conn[ai]
    s_conn = s_conn[:, ai]
    ax.imshow(s_conn, interpolation='nearest', cmap=pylab.cm.Greys)
    for i in  np.argwhere(np.diff(a[ai]) > 0):
        ax.axhline(i, c='b', alpha=0.7, linewidth=1.0)
        ax.axvline(i, c='b', alpha=0.7, linewidth=1.0)
        
    ax.set_xticks([])
    ax.set_yticks([])


    return ai

def plot_t1t1_params(fig, conn_and_dist, assign_vect, ss, hps, MAX_DIST=10, 
                     model="LogisticDistance", MAX_CLASSES = 20):
    """
    In the same order that we would plot the latent matrix, plot
    the per-parameter properties

    hps are per-relation hps

    note, tragically, this wants the whole figure

    """

    from mpl_toolkits.axes_grid1 import Grid
    from matplotlib import pylab
    
    canon_assign_vect = util.canonicalize_assignment(assign_vect)
    # create the mapping between existing and new
    canon_to_old  = {}
    for i, v in enumerate(canon_assign_vect):
        canon_to_old[v]= assign_vect[i]

    CLASSES = np.sort(np.unique(canon_assign_vect)) 
    
    CLASSN = len(CLASSES)
    print CLASSN, len(CLASSES)

    if CLASSN > MAX_CLASSES:
        print "WARNING, TOO MANY CLASSES" 
        CLASSN = MAX_CLASSES

    img_grid = Grid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (CLASSN, CLASSN),
                    axes_pad = 0.1,
                    add_all=True, 
                    share_all=True, 
                    label_mode = 'L',
                     )
    

    for c1i, c1_canon in enumerate(CLASSES[:MAX_CLASSES]):
        for c2i, c2_canon in enumerate(CLASSES[:MAX_CLASSES]):
            c1 = canon_to_old[c1_canon]
            c2 = canon_to_old[c2_canon]
            ax_pos = c1i * CLASSN + c2i
            print 'ax_pos=', ax_pos, c1i, c2i
            ax = img_grid[ax_pos]

            nodes_1 = np.argwhere(assign_vect == c1).flatten()
            nodes_2 = np.argwhere(assign_vect == c2).flatten()
            conn_dist_hist = []
            noconn_dist_hist = []
            for n1 in nodes_1:
                for n2 in nodes_2:
                    d = conn_and_dist[n1, n2]['distance']
                    if conn_and_dist[n1, n2]['link']:
                        conn_dist_hist.append(d)
                    else:
                        noconn_dist_hist.append(d)

            bins = np.linspace(0, MAX_DIST, 20)
            fine_bins = np.linspace(0, MAX_DIST, 100)

            # compute prob as a function of distance for this class
            htrue, _ = np.histogram(conn_dist_hist, bins)

            hfalse, _ = np.histogram(noconn_dist_hist, bins)

            p = htrue.astype(float) / (hfalse + htrue)
            # # TOTAL INSANE GROSS HACK REMOVE ASAP
            # pickle.dump({'conn_dist_hist' : conn_dist_hist, 
            #              'noconn_dist_hist' : noconn_dist_hist, 
            #              'htrue' : htrue, 
            #              'hfalse' : hfalse, 
            #              'p' : p, 'bins' : bins, 'fine_bins' : fine_bins}, 
            #             open("component.%d.%d.pickle" % (c1i, c2i), 'w'))
            
            ax.plot(bins[:-1], p)
            #ax.set_xlim(0, MAX_DIST)
            #ax.set_ylim(0, 1.0)
            # ax.set_xticks([])
            # ax.set_yticks([])

            if model == "LogisticDistance":
                c = ss[(c1, c2)]
                y = util.logistic(fine_bins, c['mu'], c['lambda']) 
                y = y * (hps['p_max'] - hps['p_min']) + hps['p_min']
                ax.plot(fine_bins, y, c='r') 
                ax.text(0, 0.2, r"mu: %3.2f" % c['mu'], fontsize=4)
                ax.text(0, 0.6, r"lamb: %3.2f" % c['lambda'], fontsize=4)
                ax.axvline(c['mu'], c='k')

