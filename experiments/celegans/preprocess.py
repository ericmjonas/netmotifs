import numpy as np
import cPickle as pickle
from matplotlib import pylab
from xlrd import open_workbook
import os
from ruffus import * 

DATA_DIR = "../../../data/celegans/conn2"

@files([os.path.join(DATA_DIR, x) for x in ["NeuronConnect.xls", "NeuronType.xls"]], "data.pickle")
def read_data((neuron_connect_file, neuron_type_file), output_file):


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
            
    pickle.dump({'connections' : connections, 
                 'neurons' : neurons}, 
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
    canonical_neuron_ordering = neurons.keys()
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

@transform(preprocess, suffix(".processed.pickle"), ".adjmat.pdf")
def plot_adjmat(infile, outfile):
    d = pickle.load(open(infile))
    canonical_neuron_ordering = d['canonical_neuron_ordering']
    NEURON_N = len(canonical_neuron_ordering)
    conn_matrix = d['conn_matrix']
    neurons = d['neurons']
    
    # sort by somatic position
    pos = [neurons[n]['soma_pos'] for n in canonical_neuron_ordering]
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
    ax.set_title("c. elegans. herm. somatic connectome adj. matrix")
    f.savefig(outfile)


pipeline_run([read_data, preprocess, plot_adjmat])
