import numpy as np
import cPickle as pickle
from matplotlib import pylab
from ruffus import * 
from preprocess import * 

# plot spatial distribution of each cell type
# plot area vs cell body distance

@files("countmatrix.pickle", 
       "adj_comp.pdf")
def adj_mat_plot(infile, adj_plot):

    d = pickle.load(open(infile, 'r'))
    cell_ids = d['cell_ids']
    conn = d['conn']
    f = pylab.figure()

    ax_conn = f.add_subplot(1, 1, 1)
    
    pos = []
    sizes = []
    for xi, x in enumerate(conn):
        for yi, count in enumerate(x):
            print count.shape
            if count > 0:
                pos.append((xi, yi))
                sizes.append(count)

    pos = np.array(pos)
    ax_conn.scatter(pos[:, 0], pos[:, 1], 
                    s = sizes, edgecolor='none', alpha=0.5)
    ax_conn.set_xlim(0, len(conn))
    ax_conn.set_ylim(0, len(conn))
    ax_conn.set_xlabel("presynaptic")
    ax_conn.set_ylabel("postsynaptic")

    f.savefig(adj_plot)

if __name__ == "__main__":
    pipeline_run([adj_mat_plot])
