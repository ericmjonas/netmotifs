from matplotlib import pylab

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas


import irm
import irm.data

from irm.experiments import plot_latent

def plot_cluster_properties(assignments, true_cell_types, 
                            pos_vec, synapses, output_filename,
                            cluster_thold= 0.98, 
                            class_colors=None):

    """
    Plot histogram of cell depth, 
    """


    f = pylab.figure(figsize=(18, 12))
    X_MIN = np.min(pos_vec[:, 0])
    X_MAX = np.max(pos_vec[:, 0])

    THOLDS = [1.0]
    def extra_plot_func(axs, cells_idx, col_i):
        cells = pos_vec[cells_idx]
   

        bins = np.linspace(X_MIN, X_MAX, 40)
        hist, bin_edges = np.histogram(cells[:, 0], bins=bins, normed=True)

        ax_x = axs[0]
        ax_yz = axs[1]

        ax_x.plot(hist, bins[:-1], c='b')
        ax_yz.scatter(cells[:, 1], cells[:, 2], c='b', edgecolor='none', s=5)
                
        ax_x.set_ylim(X_MAX, X_MIN)
        ax_x.set_xlim(0, 0.2)
        if col_i > 0:
            ax_x.set_yticklabels([])

        tgt_ids = cells_idx
        synapse_area_thold = THOLDS[0]
        sub_dfs = [synapses[(synapses['from_id'] == cell_id) | (synapses['to_id'] == cell_id)] for cell_id in tgt_ids]
        sub_merge_df = pandas.concat(sub_dfs)
        syn_th = sub_merge_df[sub_merge_df['area']>synapse_area_thold]
        syn_hist, syn_bin_edges = np.histogram(syn_th['x'], bins=bins, normed=True)

        ax_x.plot(syn_hist, bins[:-1], c='r', 
                  linewidth=1, alpha=0.5)
        ax_yz.scatter(syn_th['y'], syn_th['z'], c='r', s=1, edgecolor='none')
        if (col_i > 0):
            ax_yz.set_xticklabels([])
            ax_yz.set_yticklabels([])

            ax_x.set_xticklabels([])
            ax_x.set_yticklabels([])

    irm.plot.plot_purity_hists_h(f, assignments, true_cell_types, extra_rows=2, 
                                 extra_row_func=extra_plot_func, 
                                 thold = cluster_thold, class_colors = class_colors)
    #f.tight_layout()
    #f.suptitle(output_filename)
    
    f.savefig(output_filename, bbox_inches='tight')

def reorder_synapse_ids(synapse_df, cell_id_permutation):
    """
    cell_id_permutation is the permutation of the original cell IDs. 
    it takes the cell at position i and maps it to position cell_id_permutation[i]
    
    This does the same for the synapse dataframe, turning the from_id and 
    to_id 
    """

    new_df = synapse_df.copy()
    new_df['from_id'] = cell_id_permutation[synapse_df['from_id'].astype(int)]
    new_df['to_id'] = cell_id_permutation[synapse_df['to_id'].astype(int)]
    return new_df
    
