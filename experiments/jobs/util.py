import numpy as np
from matplotlib import pylab
import cPickle as pickle

zipcodeshapes = pickle.load(open("jobs/zipcodeshapes.pickle"))


def plot_zipcode_heatmap(ax, zip_vals, default_color='white', border=1):
    for zip_i, zip_row in zipcodeshapes.iterrows():
        z = int(zip_i)
        points = np.array(zip_row['points'])
        if z in zip_vals:
            v = zip_vals[z]
            if len(points) > 0:
                print points
                patch = pylab.Polygon(points , closed=True, fill=True, color=v)
                ax.add_patch(patch)
        else:
            if len(points) > 0:
                patch = pylab.Polygon(points , closed=True, fill=True, facecolor=default_color, linewidth=border, edgecolor='k')
                ax.add_patch(patch)     
