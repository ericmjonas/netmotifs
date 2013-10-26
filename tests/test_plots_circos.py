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
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(N)])
    
    cp.set_entity_links(zip(np.random.permutation(N), np.random.permutation(N)))
                 
    circos.write(cp, 'links.100.png', tempdir='plots')
    
    N = 1000
    av = np.arange(N, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(N)], 
                         label_size="10p")
    
    links = zip(np.random.permutation(N), np.random.permutation(N))
    cp.set_entity_links(links[::10])
                 
    circos.write(cp, 'links.1000.svg', tempdir='plots')
    
    
def test_ribbons():
    N = 100
    av = np.arange(N, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(N)])
    
    cp.set_class_ribbons([(0, 4), (7, 3)])
                 
    circos.write(cp, 'ribbons.100.png', tempdir='plots')

    
def test_both():
    N = 100
    av = np.arange(N, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(N)])
    cp.set_entity_links(zip(np.random.permutation(N), np.random.permutation(N)))
    
    cp.set_class_ribbons([(0, 4), (7, 3)])
                 
    circos.write(cp, 'both.100.png', tempdir='plots')
