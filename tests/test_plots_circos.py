from nose.tools import *
import numpy as np

from irm.plots import circos


def test_simple():
    # all one
    av = np.zeros(100, dtype=np.int32)
    cp = circos.CircosPlot(av)
    
    circos.write(cp, 'allone.png', tempdir='plots')

    av = np.arange(100, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    
    circos.write(cp, 'many.png', tempdir='plots')
    
def test_labels():
    av = np.arange(100, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(100)])

    circos.write(cp, 'labels.png', tempdir='plots')

def test_links():
    N = 100
    av = np.arange(N, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(100)])
    
    cp.set_entity_links(zip(np.random.permutation(N), np.random.permutation(N)))
                 
    circos.write(cp, 'links.png', tempdir='plots')
    
    
