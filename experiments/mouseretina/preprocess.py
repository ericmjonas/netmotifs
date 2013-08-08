from ruffus import *
from matplotlib import pylab
from xlrd import open_workbook
import os

import cPickle as pickle
import numpy as np
import scipy.io

LIGHT_AXIS = [0.9916,0.0572, 0.1164]

@files("../../../data/mouseretina/Helmstaedter_et_al_SUPPLinformation5.mat", 
       "rawdata.pickle")
def load_data(mat_file, output_file):
    d = scipy.io.loadmat(mat_file)
    
    data = d['kn_allContactData_Interfaces_duplCorr_output_IDconv']

    FROM_COL = 0
    TO_COL = 1 

    cell_ids = np.unique(np.vstack([data[:, FROM_COL], data[:, TO_COL]])).astype(int)
    
    CELL_N = len(cell_ids)
    
    # some cells have ZERO synapses, so they are empty; still nto sure where cellids
    # > 1123 come from 
    
    cellid_to_pos = {}
    for i, id in enumerate(cell_ids):
        cellid_to_pos[int(id)] = i

    # create the matrix
    area_mat = np.zeros((CELL_N, CELL_N), dtype=[('area', np.float32), 
                                                 ('count', np.uint32)])
    synapse_pos = {}

    for from_id, to_id, area, x, y, z in data:
        from_i = cellid_to_pos[int(from_id)]
        to_i = cellid_to_pos[int(to_id)]
        if area_mat[from_i, to_i] > 0:
            area_mat[from_i, to_i]['area'] += area
            area_mat[from_i, to_i]['count'] += 1

        if (int(from_id), int(to_id)) not in synapse_pos:
            synapse_pos[(int(from_id), int(to_id))] = []

        synapse_pos[(int(from_id), int(to_id))].append((x/1000., y/1000., z/1000.))
    pickle.dump({'area_mat' : area_mat, 
                 'synapse_pos' : synapse_pos, 
                 'cellid_to_pos' : cellid_to_pos, 
                 'cell_ids' : cell_ids}, 
                open(output_file, 'w'))

@files("../../../data/mouseretina/Helmstaedter_et_al_SUPPLinformation4.xlsx", 
       "xlsxdata.pickle")
def load_xlsx_data(xlsx_file, output_file):
    neuron_connect =  open_workbook(xlsx_file)
    connections = {}
    s =  neuron_connect.sheets()[0]
    N = 1123
    data = np.zeros((N, N), dtype=np.float32)
    for cell_i in range(N):
        for cell_j in range(N):
            print cell_i, cell_j
            syn_area = s.cell(cell_i + 1, cell_j+1).value
            data[cell_i, cell_j] = syn_area

    s = neuron_connect.sheets()[2]
    types = np.zeros(N, dtype=np.uint32)
    for cell in range(N):
        cell_id = s.cell(cell+1, 0).value
        type_id = s.cell(cell +1, 3).value
        types[cell] = type_id
    
    pickle.dump({'area_mat' : data, 
                 'types' : types}, 
                open(output_file, 'w'))

@files(load_data, ['synapse_hist.pdf'])
def sanity_check(infile, (synapse_hist,)):
    """
    Sanity checking and plotting
    """

    d = pickle.load(open(infile))
    area_mat = d['area_mat']
    synapse_pos = d['synapse_pos']

    print "MISSING", set(range(1, 1025)) - set(d['cell_ids'])

    f = pylab.figure(figsize=(8, 6))
    nonzero = area_mat.flatten()
    nonzero = nonzero[nonzero['count']>0]
    print nonzero['count']
    ax_count = f.add_subplot(1,2, 1)
    ax_count.hist(nonzero['count'], bins=20)
    ax_count.set_title("histogram of synapse counts")
    
    ax_total_area = f.add_subplot(1,2, 2)
    ax_total_area.hist(nonzero['area'], bins=20)
    ax_total_area.set_title("total area distribution")
    
    f.savefig(synapse_hist)


@files(load_data, "synapse_pos.png")
def plot_synapses(infile, outfile):
    """
    normalize positions to the inbound light
    """
    d = pickle.load(open(infile))
    area_mat = d['area_mat']
    synapse_pos = d['synapse_pos']

    all_synapses = []
    for k, x in synapse_pos.iteritems():
        all_synapses += x

    all_synapses = np.array(all_synapses)


    f = pylab.figure(figsize=(8, 8))

    alpha = 0.01
    s = 1.0
    for i in range(3):
        ax = f.add_subplot(2, 2, i+1)
        
        ax.scatter(all_synapses[:, (i) % 3], all_synapses[:, (i+1)%3], 
                   edgecolor='none', s=s, alpha=alpha, c='k')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

    f.savefig(outfile, dpi=600)

@files(load_data, "cell_adj.png")
def plot_adj(infile, outfile):
    """
    normalize positions to the inbound light
    """
    d = pickle.load(open(infile))
    area_mat = d['area_mat']
    CELL_N = len(area_mat)
    p = np.random.permutation(CELL_N)
    area_mat_p = area_mat[p, :]
    area_mat_p = area_mat_p[:, p]

    f = pylab.figure(figsize=(8, 8))
    ax = f.add_subplot(1, 1, 1)

    ax.imshow(area_mat_p['count'] > 0, interpolation='nearest', 
              cmap=pylab.cm.Greys)
    
    f.savefig(outfile, dpi=600)

pipeline_run([load_data, load_xlsx_data, sanity_check, plot_synapses, 
              plot_adj])


