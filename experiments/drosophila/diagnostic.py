import numpy as np
import cPickle as pickle
from matplotlib import pylab
from ruffus import * 
from preprocess import * 

# plot spatial distribution of each cell type
# plot area vs cell body distance

@files("data.all.pickle", 
       "pre_post_dist_hist.pdf")
def pre_post_dist_hist(infile, adj_plot):

    df = pickle.load(open(infile, 'r'))
    
    f = pylab.figure(figsize=(6, 12))
    for i, coord in enumerate(['x', 'y', 'z']):

        ax_conn = f.add_subplot(3,1, i+1)
        ax_conn.hist(df['pre.%s' % coord] - df['post.%s' % coord], bins=np.linspace(-2, 2, 100))
        ax_conn.grid()
        ax_conn.set_title("distance between pre and post %s" % coord)

    f.savefig(adj_plot)

@files("celldata.pickle", 
       "cell_locations.pdf")
def celldata_plot(infile, cell_locations):

    d = pickle.load(open(infile, 'r'))
    celldata = d['celldata']

    f = pylab.figure(figsize=(12, 6))

    cmin = {'x' : 0, 
            'y': 0, 
            'z' : 0}
    cmax = {'x' : 100, 
            'y' : 100, 
            'z' : 100}

    plot_pos = 0
    for syn in ['pre', 'post']:
        for c1, c2 in [('x', 'y'), ('y', 'z'), ('z', 'x')]:
            plot_pos += 1
            ax = f.add_subplot(2, 3, plot_pos)
            ax.scatter(celldata[syn + '.' + c1], 
                       celldata[syn + '.' + c2], edgecolor='none', alpha=0.5)
            ax.set_xlim(cmin[c1], cmax[c1])
            ax.set_ylim(cmin[c2], cmax[c2])
            
    f.savefig(cell_locations)

@files("countmatrix.pickle", 
       "adj_comp.pdf")
def adj_mat_plot(infile, adj_plot):

    d = pickle.load(open(infile, 'r'))
    cell_ids = np.array(d['cell_ids'])
    cell_id_pos = np.argsort(cell_ids)
    conn = d['conn']
    print len(conn)
    conn = conn[cell_id_pos]
    conn = conn[:, cell_id_pos]

    f = pylab.figure(figsize=(6,6))

    ax_conn = f.add_subplot(1, 1, 1)
    
    pos = []
    sizes = []
    for xi, x in enumerate(conn):
        for yi, count in enumerate(x):
            if count > 0:
                pos.append((xi, yi))
                sizes.append(count)

    pos = np.array(pos)
    ax_conn.scatter(pos[:, 1], pos[:, 0], 
                    s = np.array(sizes)*3, edgecolor='none', alpha=0.3, c='k')
    ax_conn.set_xlim(0, len(conn))
    ax_conn.set_ylim(len(conn), 0)
    ax_conn.set_ylabel("presynaptic")
    ax_conn.set_xlabel("postsynaptic")
    ax_conn.set_title("Drosophila optic medulla")
    f.tight_layout()
    f.savefig(adj_plot)

if __name__ == "__main__":
    pipeline_run([adj_mat_plot, celldata_plot, pre_post_dist_hist])
