import numpy as np
from matplotlib import pylab
import irm

def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


SIDE_N = 5

# class_conn = {(0, 1) : ('d', 1.0, 0.7), 
#               (1, 2) : ('d', 2.0, 0.8), 
#               (3, 2) : ('p', 0.1), 
#               (3, 0) : ('d', 1.7, 0.9), 
#               (1, 3) : ('p', 0.1)}


# nodes_with_class, connectivity = irm.data.generate.c_mixed_dist_block(SIDE_N, 
#                                                                       class_conn, JITTER=0.1, rand_conn_prob=0.01)


# class_conn = {(0, 1) : (1.0, 0.8, 0.1), 
#               (1, 2) : (2.0, 0.4, 0.3), 
#               (3, 2) : (0.7, 0.9, 0.2), 
#               (3, 0) : (1.7, 0.7, 0.5), 
#               (1, 3) : (0.5, 0.5, 0.2)}


# nodes_with_class, connectivity = irm.data.generate.c_bump_dist_block(SIDE_N, 
#                                                                       class_conn, JITTER=0.3, rand_conn_prob=0.01)



class_conn = {(0, 1) : (1.0, 0.8), 
              (1, 2) : (2.0, 0.6), 
              (3, 2) : (0.7, 0.9), 
              (2, 1) : (0.7, 0.8), 
              (0, 3) : (1.7, 0.7)}


nodes_with_class, connectivity = irm.data.generate.c_class_neighbors(SIDE_N, 
                                                                     class_conn, JITTER=0.3, rand_conn_prob=0.01)



CELL_N = len(connectivity)

conn_and_dist = np.zeros((CELL_N, CELL_N), 
                         dtype = [('link', np.uint8), 
                                  ('distance', np.float32)])

for ni, (ci, posi) in enumerate(nodes_with_class):
    for nj, (cj, posj) in enumerate(nodes_with_class):
        conn_and_dist[ni, nj]['link'] = connectivity[ni, nj]
        conn_and_dist[ni, nj]['distance'] = dist(posi, posj)



f = pylab.figure()
ax = f.add_subplot(1, 1, 1)
ai = np.random.permutation(CELL_N)
c_r = connectivity[ai, :]
c_r = c_r[:, ai]

ax.imshow(c_r, interpolation='nearest', cmap=pylab.cm.Greys)

c_class = nodes_with_class['class']
ai = np.argsort(c_class).flatten()
c_sorted = connectivity[ai, :]
c_sorted = c_sorted[:, ai]
f.savefig("f1.raw.pdf")

f2 = pylab.figure()
ax = f2.add_subplot(1, 1, 1)

c_class_sorted = c_class[ai]
di = np.argwhere(np.diff(c_class_sorted) > 0).flatten()
ax.imshow(c_sorted, interpolation='nearest', cmap=pylab.cm.Greys)
for d in di:
    ax.axhline(d+0.5)
    ax.axvline(d+0.5)
f2.savefig("f1.sorted.pdf")

f3 = pylab.figure()
irm.plot.plot_t1t1_params(f3, conn_and_dist, nodes_with_class['class'], 
                          None, None, model=None, MAX_DIST=4)
f3.savefig("f1.latent.pdf")

