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
    
    cp.set_class_ribbons([(0, 4, 2), (7, 3, 2)])
                 
    circos.write(cp, 'ribbons.100.png', tempdir='plots')

def test_multi_ribbons():
    N = 100
    av = np.arange(N, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(N)])
    
    cp.add_class_ribbons([(0, 4, 2), (7, 3, 2)], 'red_a5')
    cp.add_class_ribbons([(2, 3, 6), (1, 4, 4)], 'blue_a5')
                 
    circos.write(cp, 'ribbons.multi.100.png', tempdir='plots')

    
def test_both():
    N = 100
    av = np.arange(N, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    cp.set_entity_labels(["conn%d" % x for x in  np.arange(N)])
    cp.set_entity_links(zip(np.random.permutation(N), np.random.permutation(N)))
    
    cp.set_class_ribbons([(0, 4, 2), (7, 3, 4)])
                 
    circos.write(cp, 'both.100.png', tempdir='plots')

def test_scatter_plot():
    # all one
    av = np.arange(100, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    data_points = av
    
    cp.add_plot('scatter', {'r0' : '1.1r', 
                            'r1' : '1.2r', 
                            'min' : 0, 
                            'max' : 10}, 
                data_points, 
                {'backgrounds' : [('background', {'color': 'vvlgreen', 
                                                  'y0' : 4, 
                                                  'y1' : 8})]}
                )
    
    circos.write(cp, 'plots.scatter.png', tempdir='plots')

def test_heatmap_plot():
    # all one
    av = np.arange(100, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    data_points = av
    
    cp.add_plot('heatmap', {'r0' : '1.1r', 
                            'r1' : '1.2r', 
                            'min' : 0, 
                            'max' : 10}, 
                data_points, 
                )
    
    circos.write(cp, 'plots.heatmap.png', tempdir='plots')

def test_glyph_plot():
    # all one
    av = np.arange(300, dtype=np.int32) % 10
    cp = circos.CircosPlot(av)
    data_points = ["C" for d in av] 
    for i in range(0, 300, 7):
        data_points[i]= 'A'
    for i in range(0, 300, 4):
        data_points[i]= 'L'
    
    cp.add_plot('text', {'r0' : '1.05r', 
                         'r1' : '1.10r', 
                         'label_size' : '20p', 
                         'label_font' : 'glyph', 
                         'label_rotate' : 'yes', 
                     }, 
                
                data_points)
    
    circos.write(cp, 'plots.glyphs.png', tempdir='plots')
