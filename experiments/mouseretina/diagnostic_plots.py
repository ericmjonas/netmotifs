import numpy as np
import cPickle as pickle
from matplotlib import pylab
from ruffus import * 
from preprocess import * 

# plot spatial distribution of each cell type
# plot area vs cell body distance

@files(["conn.areacount.pickle", "xlsxdata.pickle", 'soma.positions.pickle'], 
       "adj_comp.pdf")
def mat_xls_consistency((conn_areacount, xlsdata, somapos), adj_plots):
    """
    To what degree does the excel data, the matlab data, and the
    soma position data agree? 

    """
    xls_d = pickle.load(open(xlsdata, 'r'))
    mat_d = pickle.load(open(conn_areacount, 'r'))
    soma_d = pickle.load(open(somapos, 'r'))

    print "CELLS IN XLS:", len(xls_d['area_mat'])
    print "CELLS IN MATLAB:", len(mat_d['area_mat'])
    print "CELLS IN SOMA IMAGES:", len(soma_d['pos_vec'])
    CELL_N = len(soma_d['pos_vec'])
    xls_m = xls_d['area_mat'][:CELL_N, :CELL_N]
    mat_m = mat_d['area_mat'][:CELL_N, :CELL_N]['area']
    
    f = pylab.figure()
    ax_delta = f.add_subplot(1, 1, 1)
    delta = np.abs(xls_m - mat_m)
    delta_signed = xls_m - mat_m
    print "There are", np.sum(delta > 0.001), "disagreeing cells"
    for delta_pos in np.argwhere(delta > 0.001):
        p0 = delta_pos[0]
        p1 = delta_pos[1]
        print "at ", delta_pos, "the error is", delta_signed[p0, p1]
        sc = mat_d['area_mat'][p0, p1]['count']
        print "synapse count =", sc
            
    ax_delta.hist(delta_signed, bins=20)
    ax_delta.set_xlabel("spreadsheet - mat")
    ax_delta.set_ylabel("count")
    ax_delta.set_title("Difference between xls and mat")

    f.savefig(adj_plots)
    
    
# @files(load_data, ['synapse_hist.pdf'])
# def sanity_check(infile, (synapse_hist,)):
#     """
#     Sanity checking and plotting
#     """

#     d = pickle.load(open(infile))
#     area_mat = d['area_mat']
#     synapse_pos = d['synapse_pos']

#     print "MISSING", set(range(1, 1025)) - set(d['cell_ids'])

#     f = pylab.figure(figsize=(8, 6))
#     nonzero = area_mat.flatten()
#     nonzero = nonzero[nonzero['count']>0]
#     print nonzero['count']
#     ax_count = f.add_subplot(1,2, 1)
#     ax_count.hist(nonzero['count'], bins=20)
#     ax_count.set_title("histogram of synapse counts")
    
#     ax_total_area = f.add_subplot(1,2, 2)
#     ax_total_area.hist(nonzero['area'], bins=20)
#     ax_total_area.set_title("total area distribution")
    
#     f.savefig(synapse_hist)


@files("synapses.pickle", "synapse_pos.png")
def plot_synapses(infile, outfile):
    """
    """
    d = pickle.load(open(infile))
    df = d['synapsedf']

    f = pylab.figure(figsize=(16, 8))

    alpha = 0.05
    s = 1.0
    ax_xz = f.add_subplot(2, 1, 1)
    ax_xz.scatter(df['z'], df['x'], 
                  edgecolor='none', s=df['area']*4, alpha=alpha, c='k')
    ax_xz.set_xlim(0, MAX_DIM[2])
    ax_xz.set_xlabel("z (um)")
    ax_xz.set_ylim(0, MAX_DIM[0])
    ax_xz.set_ylabel("x (um)")

    ax_yz = f.add_subplot(2, 1, 2)
    ax_yz.scatter(df['y'], df['x'], 
                  edgecolor='none', s=df['area']*4, alpha=alpha, c='k')
    ax_yz.set_xlim(0, MAX_DIM[1])
    ax_yz.set_xlabel("y (um)")
    ax_yz.set_ylim(0, MAX_DIM[0])
    ax_yz.set_ylabel("x (um)")

    f.savefig(outfile, dpi=600)

@files("conn.areacount.pickle", "cell_adj.png")
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

@files(["type_metadata.pickle", "soma.positions.pickle"], 
       ['somapos.png'])
def plot_somapos((type_file, pos_file), (pos_outfile,)):
    soma_pos = pickle.load(open(pos_file, 'r'))
    type_metadata = pickle.load(open(type_file, 'r'))['type_metadata']

    pos_vec = soma_pos['pos_vec']

    f = pylab.figure(figsize=(16, 8))
    ax = f.add_subplot(2, 1, 1)
    ax.scatter(pos_vec[:, 1], pos_vec[:, 0])

    ax = f.add_subplot(2, 1, 2)
    
    ax.scatter(pos_vec[:, 2], pos_vec[:, 0])

    f.savefig(pos_outfile)
    
pipeline_run([mat_xls_consistency, plot_synapses, plot_adj, plot_somapos])
