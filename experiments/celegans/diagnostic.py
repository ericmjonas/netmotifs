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
        ax.set_ylim(0, NEURON_N)
        ax.set_title("c. elegans. herm. somatic connectome adj. matrix %s" % sortby)
        f.savefig(outfile)

@files("data.processed.pickle", 'data.class.pdf')
def plot_classmat(infile, outfile):
    d = pickle.load(open(infile))
    canonical_neuron_ordering = d['canonical_neuron_ordering']
    NEURON_N = len(canonical_neuron_ordering)

    conn_matrix = d['conn_matrix']
    neurons = d['neurons']

    pos = np.array(neurons['class'])
    sorted_pos = np.argsort(pos)

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
    

    f = pylab.figure(figsize=(24, 24))
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
    f.savefig(outfile)


pipeline_run([plot_adjmat, plot_classmat, plot_positions])
