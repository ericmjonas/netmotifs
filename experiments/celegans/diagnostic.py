from ruffus import *
import cPickle as pickle
import numpy as np
import pandas
import copy
import os, glob
import time
from matplotlib import pylab
from jinja2 import Template
import irm
import irm.data
import matplotlib.gridspec as gridspec
from matplotlib import colors
import copy


@files('data.processed.pickle', 'class.positions.pdf')
def plot_positions(infile, outfile):
    
    data = pickle.load(open(infile, 'r'))
    n = data['neurons']
    
    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    plotted = 0
    for ei, (gi, g) in enumerate(n.groupby('class')):
        N = len(g)
        if N >= 4:
            r = g.iloc[0]['role']
            if r == 'M':
                c = 'b'
            elif r == 'S':
                c = 'g'
            else:
                c = "#AAAAAA"
            ax.plot([0, 1], [plotted, plotted], c='k', alpha=0.5, linewidth=1)
            ax.scatter(g['soma_pos'], np.ones(N) * plotted, 
                       c=c)
            ax.text(1.02, plotted, gi, fontsize=8)
            plotted +=1
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-1, plotted)
    ax.set_title('c elegans cell class positions')
    f.savefig(outfile)
    
@files('data.processed.pickle', 'data.pos.pdf')
def plot_adjmat(infile, plot_pos_filename):
    d = pickle.load(open(infile))
    canonical_neuron_ordering = d['canonical_neuron_ordering']
    NEURON_N = len(canonical_neuron_ordering)

    conn_matrix = d['conn_matrix']
    neurons = d['neurons']
    print "NEURON_N=", NEURON_N, conn_matrix.shape    
    # sort by somatic position
    for sortby, outfile in [('soma_pos', plot_pos_filename)]:
        pos = neurons['soma_pos']
        sorted_pos = np.argsort(pos)
        conn_matrix = conn_matrix[sorted_pos]
        conn_matrix = conn_matrix[:, sorted_pos]

        chem_points = []
        elec_points = []
        for n1_i in range(NEURON_N):
            for n2_i in range(NEURON_N):
                if conn_matrix[n1_i, n2_i]['chemical'] > 0:
                    chem_points.append((n1_i, n2_i, conn_matrix[n1_i, n2_i]['chemical']))
                if conn_matrix[n1_i, n2_i]['electrical'] > 0:
                    elec_points.append((n1_i, n2_i, conn_matrix[n1_i, n2_i]['electrical']))

        chem_points = np.array(chem_points)
        elec_points = np.array(elec_points)


        f = pylab.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.scatter(chem_points[:, 0], chem_points[:, 1], c='b',
                   edgecolor='none', s= chem_points[:, 2]*3, alpha=0.5)
        ax.scatter(elec_points[:, 0], elec_points[:, 1], c='r', 
                   edgecolor='none', s= elec_points[:, 2]*3, alpha=0.5)
        ax.set_xlim(0, NEURON_N)
        ax.set_ylim(NEURON_N, 0)
        ax.set_title("c. elegans. herm. somatic connectome adj. matrix %s" % sortby)
        f.tight_layout()
        f.savefig(outfile)

@files("data.processed.pickle", ['data.all.class.pdf', 'data.big.class.pdf'])
def plot_classmat(infile, (all_out, big_out)):
    d = pickle.load(open(infile))

    conn_matrix = d['conn_matrix']
    n = d['neurons']

    for class_size, outfile in [(1, all_out), (4, big_out)]:
        sizes = n.groupby('class').size()
        large_classes = sizes[sizes >= class_size].index
        which = n['class'].apply(lambda x : x in large_classes)
        neurons = n[which]

        pos = np.array(neurons['class'])
        sorted_pos = np.argsort(pos)
        NEURON_N = len(neurons)
        conn_matrix = conn_matrix[which]
        conn_matrix = conn_matrix[:, which]

        conn_matrix = conn_matrix[sorted_pos]
        conn_matrix = conn_matrix[:, sorted_pos]

        chem_points = []
        elec_points = []


        c = np.array(neurons['class'])
        c = c[np.argsort(c)]
        boundaries = []
        for i, ci in enumerate(c[:-1]):
            if ci != c[i+1]:
                boundaries.append(i)


        f = pylab.figure(figsize=(48, 48))
        ax = f.add_subplot(1, 1, 1)

        cmap = colors.ListedColormap(['white', 'red', 'blue', 
                                      'purple'])
        bounds=[0,1, 2, 3, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        data_mat = (conn_matrix['chemical'] > 0).astype(int)
        data_mat +=(conn_matrix['electrical'] > 0).astype(int) *2

        ax.imshow(data_mat, 
                  interpolation='nearest', cmap=cmap, norm=norm)


        for b in boundaries:
            ax.axhline(b+0.5, c='k', linewidth=0.1, alpha=0.5)
            ax.axvline(b+0.5, c='k', linewidth=0.1, alpha=0.5)

        ax.set_xlim(0, NEURON_N)
        ax.set_ylim(0, NEURON_N)
        ax.set_title("c. elegans. herm. somatic connectome adj. matrix class")
        
        ax.set_yticks([])
        ax.set_yticks(np.arange(NEURON_N) + 0.5, minor=True)
        ax.set_yticklabels(c, minor=True)

        ax.set_xticks([])
        ax.set_xticks(np.arange(NEURON_N) + 0.5, minor=True)
        ax.set_xticklabels(c, minor=True, rotation=90)
        ax.xaxis.set_label_position("top")
        f.savefig(outfile)

@files("data.processed.pickle", ['data.all.dist.pdf', 'data.med.dist.pdf', 'data.big.dist.pdf'])
def plot_conn_dist(infile, (all_out, med_out, big_out)):
    d = pickle.load(open(infile))

    conn_matrix = d['conn_matrix']
    n = d['neurons']
    MAX_DIST = 1.0
    bins = np.linspace(0, MAX_DIST, 20)

    for class_size, outfile in [(2, all_out), (4, med_out),  (6, big_out)]:
        sizes = n.groupby('class').size()
        large_classes = sizes[sizes >= class_size].index
        which = n['class'].apply(lambda x : x in large_classes)
        neurons = n[which]

        class_prob_dists = {'chemical' : {}, 
                            'electrical' : {}}

        for class_1 in large_classes:
            for class_2 in large_classes:

                chemical_conn_dist_hist = []
                chemical_noconn_dist_hist = []
                electrical_conn_dist_hist = []
                electrical_noconn_dist_hist = []

                
                nodes_1 = np.argwhere(n['class'] == class_1).flatten()
                nodes_2 = np.argwhere(n['class'] == class_2).flatten()
                for n1 in nodes_1:
                    for n2 in nodes_2:
                        sp_1 = n.iloc[n1]['soma_pos'] 
                        sp_2 = n.iloc[n2]['soma_pos']
                        d = np.abs(sp_1 - sp_2)

                        if conn_matrix[n1, n2]['chemical']  > 0:
                            chemical_conn_dist_hist.append(d)
                        else:
                            chemical_noconn_dist_hist.append(d)

                        if conn_matrix[n1, n2]['electrical']  > 0:
                            electrical_conn_dist_hist.append(d)
                        else:
                            electrical_noconn_dist_hist.append(d)


                htrue, _ = np.histogram(chemical_conn_dist_hist, bins)
                hfalse, _ = np.histogram(chemical_noconn_dist_hist, bins)
                p = htrue.astype(float) / (hfalse + htrue)
                class_prob_dists['chemical'][(class_2, class_1)] = p

                htrue, _ = np.histogram(electrical_conn_dist_hist, bins)
                hfalse, _ = np.histogram(electrical_noconn_dist_hist, bins)
                p = htrue.astype(float) / (hfalse + htrue)
                class_prob_dists['electrical'][(class_2, class_1)] = p
                # debgu
                if class_1 == "IL1" and class_2 == "IL1":
                    print class_prob_dists['chemical'][(class_2, class_1)]
                    print class_prob_dists['electrical'][(class_2, class_1)]
                    

                # prob_con_matrix = {}
                

        
        pos = np.array(neurons['class'])
        sorted_pos = np.argsort(pos)
        NEURON_N = len(neurons)
        conn_matrix = conn_matrix[which]
        conn_matrix = conn_matrix[:, which]

        conn_matrix = conn_matrix[sorted_pos]
        conn_matrix = conn_matrix[:, sorted_pos]

        chem_points = []
        elec_points = []


        c = np.array(neurons['class'])
        c = c[np.argsort(c)]
        boundaries = []
        class_start_pos = {}
        for i, ci in enumerate(c[:-1]):
            if ci != c[i+1]:
                boundaries.append(i)
                class_start_pos[ci] = i
                
                
        class_boundaries = irm.util.get_boundaries(c)
        
        f = pylab.figure(figsize=(48, 48))
        ax = f.add_subplot(1, 1, 1)


        cmap = colors.ListedColormap(['white', 'red', 'blue', 
                                      'purple'])
        bounds=[0,1, 2, 3, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        data_mat = (conn_matrix['chemical'] > 0).astype(int)
        data_mat +=(conn_matrix['electrical'] > 0).astype(int) *2

        ax.imshow(data_mat, 
                  interpolation='nearest', cmap=cmap, norm=norm)
        
        # now the classes
        cl = np.unique(c)
        cl = sorted(cl)
        from  matplotlib.patches import Rectangle
        for cl1 in cl:
            for cl2 in cl:
                p_e = class_prob_dists['electrical'][(cl1, cl2)]
                p_c = class_prob_dists['chemical'][(cl1, cl2)]
                
                b1 = class_boundaries[cl1]
                b2 = class_boundaries[cl2]
                w = b1[1]-b1[0]
                h = b2[1] - b2[0]
                x_pos = b1[0] - 0.5
                y_pos = b2[0] - 0.5
                ax.add_patch(Rectangle((x_pos, y_pos), w, h, 
                                       alpha=0.1, linewidth=2, 
                                       facecolor='none'))
                t = bins[:-1]
                scaled_t = t/np.max(t) * w  + x_pos
                
                scaled_y = p_e * h*0.9 + y_pos
                pylab.plot(scaled_t, scaled_y, c='b', linewidth=4)
                pylab.scatter(scaled_t, scaled_y, c='b', 
                              s=40, edgecolor='none')
                scaled_y = p_c * h + y_pos
                pylab.plot(scaled_t, scaled_y, c='r', linewidth=4)
                pylab.scatter(scaled_t, scaled_y, c='r', 
                              s=40, edgecolor='none')
 
        # for b in boundaries:
        #     ax.axhline(b+0.5, c='k', linewidth=0.1, alpha=0.5)
        #     ax.axvline(b+0.5, c='k', linewidth=0.1, alpha=0.5)

        ax.set_title("c. elegans. herm. somatic connectome adj. matrix class")
        
        ax.set_yticks([])
        ax.set_yticks(np.arange(NEURON_N), minor=True)
        ax.set_yticklabels(c, minor=True)

        ax.set_xticks([])
        ax.set_xticks(np.arange(NEURON_N), minor=True)
        ax.set_xticklabels(c, minor=True, rotation=90)
        ax.xaxis.set_label_position("top")

        ax.set_xlim(-0.5, NEURON_N - 0.5)
        ax.set_ylim(-0.5, NEURON_N - 0.5)

        f.savefig(outfile)


pipeline_run([plot_adjmat, plot_classmat, plot_positions, 
              plot_conn_dist])
