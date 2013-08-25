from matplotlib import pylab

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


import irm
import irm.data

from irm.experiments import plot_latent

def plot_cluster_properties(cell_list, pos_vec, synapse_df):
    """
    Plot histogram of cell depth, 
    """
