import numpy as np
from matplotlib import pylab
import irm

def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))

np.random.seed(0)

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
                                                                     class_conn, JITTER=0.3, default_param=0.01)



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
f.savefig("source.f1.raw.pdf")

f2 = pylab.figure()
ax = f2.add_subplot(1, 1, 1)

c_class_sorted = c_class[ai]
di = np.argwhere(np.diff(c_class_sorted) > 0).flatten()
ax.imshow(c_sorted, interpolation='nearest', cmap=pylab.cm.Greys)
for d in di:
    ax.axhline(d+0.5)
    ax.axvline(d+0.5)
f2.savefig("source.f1.sorted.pdf")

# hilariously construct suffstats and hps by hand
hps = {'mu_hp' : 1.0, 
       'lambda_hp' : 1.0, 
       'p_max' : 0.9, 
       'p_min' : 0.0001}
ss = {}
for c1 in range(4):
    for c2 in range(4):
        c = (c1, c2)
        if c in class_conn:
            mu = class_conn[c][0]
            lamb = class_conn[c][0]/8
        else:
            mu = 0.0001
            lamb = 0.0001

        ss[c] = {'mu' : mu, 'lambda' : lamb}


f3 = pylab.figure()
irm.plot.plot_t1t1_params(f3, conn_and_dist, nodes_with_class['class'], 
                          ss, hps, model="LogisticDistance", MAX_DIST=3.5)
f3.savefig("source.f1.latent.pdf")

DISTS = [0.1, 1.0, 2.5]
for dist_i, dist_threshold in enumerate(DISTS):
    circos_p = irm.plots.circos.CircosPlot(c_class)

    v = irm.irmio.latent_distance_eval(dist_threshold, 
                                       ss, hps, 'LogisticDistance')

    thold = 0.5
    ribbons = []
    links = []
    pairs_plotted = set()

    ribbons = []
    for (src, dest) in v.keys():
        p1 = v[(src, dest)]
        p2 = v[(dest, src)]
        p = max(p1, p2)
        if (src, dest) in pairs_plotted or (dest, src) in pairs_plotted:
            pass
        else:
            if p > thold :
                pix = int(10*p)
                print src, dest, p, pix

                ribbons.append((src, dest, pix))

        pairs_plotted.add((src, dest))

    circos_p.add_class_ribbons(ribbons)

    irm.plots.circos.write(circos_p, "source.f1.circos.%d.png" % dist_i)




