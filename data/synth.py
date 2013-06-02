import numpy as np
from matplotlib import pylab
import cPickle as pickle


def d(x, y):
    return np.sqrt(np.sum((x - y)**2))

def example():
    """
    Create an example graph with the indicated latent classes and the location-specific
    connectivity

    # there are three graph classes, and node i connects to node J if: 
    # class
    

    """
    CLASSN = 4

    # create the cells in grids; 
    GRID_N = 16
    DIST_X = 1.0
    DIST_Y = 1.0
    DIST_Z = 1.0
    nodes = np.zeros(CLASSN * GRID_N * GRID_N, dtype=[('class',  np.uint32), 
                                                      ('pos' ,  np.float32, (3, ))])

    NODEN  = len(nodes)
    
    ni = 0
    for c in range(CLASSN):
        for xi in range(GRID_N):
            for yi in range(GRID_N):
                x = xi * DIST_X
                y = yi * DIST_Y
                z = c * DIST_Z
                nodes[ni]['class'] = c
                nodes[ni]['pos'][:] = (x, y, z)
                ni += 1

    # BAD IDEA but whatever: wire things up which is horribly N^2
    def node_pred(n1, n2):
        c1 = n1['class']
        pos1 = n1['pos']
        c2 = n2['class']
        pos2 = n2['pos']

        p = 0.001
        # if c1 == 0 and c2 == 1:
        #     if d(pos1, pos2) < 4:
        #         p = 0.4
        # elif c1 == 1 and c2 == 2:
        #     if d(pos1, pos2) > 3 and d(pos1, pos2) < 6:
        #         p = 0.2
        # elif c1 == 2 and c2 == 3:
        #     p = 0.05
        # elif c1 == 3 and c2 == 1:
        #     p = max(1.0 - d(pos1, pos2) / 5., 0.0)
        if c1 == 0 and c2 == 1:
            p = 0.4
        elif c1 == 1 and c2 == 2:
            p = 0.2
        elif c1 == 2 and c2 == 3:
            p = 0.05
        elif c1 == 3 and c2 == 1:
            p=0.7
        return np.random.rand() < p

    connectivity = np.zeros((NODEN, NODEN), dtype=np.bool)
    for ni in range(NODEN):
        for nj in range(NODEN):
            connectivity[ni, nj] = node_pred(nodes[ni], nodes[nj])
    
    return  nodes, connectivity

def generate():
    np.random.seed(0)
    nodes, connectivity = example()
    NODEN = len(nodes)
    print connectivity.shape
    pylab.subplot(1, 2, 1)
    pylab.imshow(connectivity, cmap=pylab.cm.binary, interpolation='nearest')
    c = connectivity.copy()
    c = c[np.random.permutation(NODEN)]
    c = c[:, np.random.permutation(NODEN)]
    pylab.subplot(1, 2, 2)
    pylab.imshow(c, cmap=pylab.cm.binary, interpolation='nearest')

    pylab.show()

    pickle.dump({'nodes' : nodes, 
                 'connectivity' : connectivity}, 
                open('data.pickle', 'w'))

if __name__ == "__main__":
    generate()
