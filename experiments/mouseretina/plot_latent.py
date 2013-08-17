import numpy as np
import util
import cPickle as pickle 
import sys


if __name__ == "__main__":
    data_filename = sys.argv[1]
    latent_filename = sys.argv[2]
    outfile = sys.argv[3]

    data = pickle.load(open(data_filename, 'r'))
    latent = pickle.load(open(latent_filename, 'r'))
    dist_matrix = data['relations']['R1']['data']

    util.plot_latent(latent, dist_matrix, outfile, PLOT_MAX_DIST=100., 
                     MAX_CLASSES=20)

