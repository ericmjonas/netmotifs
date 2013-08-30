import numpy as np
import cPickle as pickle
from matplotlib import pylab
from xlrd import open_workbook
import pandas
import os
from ruffus import * 
from matplotlib import colors

DATA_DIR = "../../../data/celegans/conn2"

@files([os.path.join(DATA_DIR, x) for x in ["NeuronConnect.xls", "NeuronType.xls", '../manualmetadata.xlsx']], "data.pickle")
def read_data((neuron_connect_file, neuron_type_file, 
               manual_metadata_file), output_file):


    neuron_connect =  open_workbook(neuron_connect_file)
    connections = {}
    for s in neuron_connect.sheets():
        for row_i in range(1, s.nrows):
            neuron_1 = s.cell(row_i, 0).value
            neuron_2 = s.cell(row_i, 1).value
            typ = s.cell(row_i, 2).value
            nbr = int(s.cell(row_i, 3).value)

            c = neuron_1, neuron_2
            if c not in connections:
                connections[c] = []
            connections[c].append((typ, nbr))

    neurons = {}
    neuron_connect =  open_workbook(neuron_type_file)
    for s in neuron_connect.sheets():

        for row_i in range(1, s.nrows):
            neuron = str(s.cell(row_i, 0).value)
            soma_pos = float(s.cell(row_i, 1).value)
            neurons[neuron] = {'soma_pos' : soma_pos}

    # now open the excel workbook of our metadata
    metadata_df = pandas.io.excel.read_excel(manual_metadata_file, 
                                             'properties', 
                                             index_col=0)
    assert len(metadata_df) == len(neurons)
    ndf = pandas.Series([d['soma_pos'] for d in neurons.values()], 
                        neurons.keys())
    metadata_df['soma_pos'] = ndf
    n = metadata_df

    # find classes
    classes = {}
    orig_names = set(n.index)
    orig_names_base = set(orig_names)
    for cur_n in orig_names_base:
        #cur_n = orig_names.pop()
        # does its name end in two digits? That's your class

        if cur_n not in orig_names:
            continue
        if cur_n[-2:].isdigit():
            base = cur_n[:-2] 
            tgts = [s for s in orig_names if s.startswith(base) and s[-2:].isdigit()]
            classes[base] = tgts
            orig_names -= set(tgts)

    orig_names_base = set(orig_names)
    for cur_n in orig_names_base:
        if cur_n not in orig_names:
            continue
        if cur_n[-2:] == 'DL':
            base = cur_n[:-2]
            print "base =", base
            if base+"VL" in orig_names:
                # is this six way? 
                tgts = [base + sub for sub in ['DL', 'DR', 'VL', 'VR', 'L', 'R']]
                if set(tgts).issubset(orig_names):
                    # we are good to go
                    classes[base] = tgts
                    orig_names -= set(tgts)
                # four way
                tgts = [base + sub for sub in ['DL', 'DR', 'VL', 'VR']]
                if set(tgts).issubset(orig_names):
                    # we are good to go
                    classes[base] = tgts
                    orig_names -= set(tgts)
                

    orig_names_base = set(orig_names)
    for cur_n in orig_names_base:
        if cur_n not in orig_names:
            continue
        if cur_n[-1] == 'R' and cur_n[-2] != 'V':
            base = cur_n[:-1]
            if base+"L" in orig_names:
                tgts = [base + 'R', base+'L']
                classes[base] = tgts
                orig_names -= set(tgts)
    # the singletons
    for o in orig_names:
        classes[o] = [o]
    print 'classes', classes['IL1']

    n_s = []
    c_s = []
    for c, ns in classes.iteritems():
        for neuron in ns:
            n_s.append(neuron)
            c_s.append(c)
    s= pandas.Series(c_s, index=n_s)
    n['class']=s        

    pickle.dump({'connections' : connections, 
                 'neurons' : n}, 
                open(output_file, 'w'))

@transform(read_data, suffix(".pickle"), ".processed.pickle")
def preprocess(infile, outfile):
    """ 
    Construct the connectivity matrix
    """
    
    d = pickle.load(open(infile))
    connections = d['connections']

    neurons = d['neurons']
    print "THERE ARE", len(neurons), "NEURONS" 

    sizes = neurons.groupby('class').size()
    large_classes = sizes[sizes >= 4].index
    large_df = neurons[neurons['class'].apply(lambda x : x in large_classes)]
    neurons = large_df

    canonical_neuron_ordering = np.array(neurons.index)
    NEURON_N = len(canonical_neuron_ordering)
    conn_matrix = np.zeros((NEURON_N, NEURON_N), 
                           dtype = [('chemical', np.int32), 
                                    ('electrical', np.int32)])
    for n1_i, n1 in enumerate(canonical_neuron_ordering):
        for n2_i, n2 in enumerate(canonical_neuron_ordering):
            c = (n1, n2)
            if c in connections:
                for synapse_type, nbr in connections[c]:
                    code = synapse_type[0]
                    if code == 'S':
                        conn_matrix[n1_i, n2_i]['chemical'] = nbr
                        
                    elif code == 'S':
                        conn_matrix[n2_i, n1_i]['chemical'] = nbr
                        
                    elif code == 'E':
                        conn_matrix[n1_i, n2_i]['electrical'] = nbr
                    elif code == "N":
                        print 'NMJ what do I do with this', nbr, c
                # decoding: 
                # (S)end (n1 presymaptic to n2)
                # (R)eceive (n1 postsynaptic to n2)
                # (E)lectrical synapse
                # (N)MJ neuromuscular junction

        
    pickle.dump({'canonical_neuron_ordering' : canonical_neuron_ordering, 
                 'conn_matrix' : conn_matrix, 
                 'connections' : connections, 
                 'neurons' : neurons}, 
                open(outfile, 'w'))

@transform(preprocess, suffix(".processed.pickle"), [".adjmat.pos.pdf",])
def plot_adjmat(infile, (plot_pos_filename,)):
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

@transform(preprocess, suffix(".processed.pickle"), [".adjmat.class.pdf",])
def plot_classmat(infile, (outfile,)):
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


pipeline_run([read_data, preprocess, plot_adjmat, plot_classmat])
