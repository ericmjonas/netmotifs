from ruffus import *
from matplotlib import pylab
from xlrd import open_workbook
import glob
import os
import pandas


import cPickle as pickle
import numpy as np
import scipy.io
import skimage 
import skimage.draw
import skimage.feature
import skimage.morphology
import skimage.measure
import skimage.io

LIGHT_AXIS = [0.9916,0.0572, 0.1164]

@files("../../../data/mouseretina/Helmstaedter_et_al_SUPPLinformation5.mat", 
       "synapses.pickle")
def load_synapse_data(mat_file, output_file):
    """
    raw synapse data from the matlab file. 
    zero-center
    """
    
    d = scipy.io.loadmat(mat_file)
    
    data = d['kn_allContactData_Interfaces_duplCorr_output_IDconv']

    synapses = {}
    for from_id, to_id, area, x, y, z in data:
        # ASSUMPTION : from < to, upper right-hand of matrix
        from_id -= 1
        to_id -= 1
        if from_id > to_id:
            from_id, to_id = to_id, from_id
        node_tuple = (int(from_id), int(to_id)) 

        if node_tuple not in synapses:
            synapses[node_tuple] = []

        synapses[node_tuple].append(((x/1000., y/1000., z/1000.), area))
    pickle.dump({'synapses' : synapses},  
                 open(output_file, 'w'))

@files("../../../data/mouseretina/Helmstaedter_et_al_SUPPLinformation5.mat", 
       "rawdata.pickle")
def load_data(mat_file, output_file):
    """
    In a move that may doom us all to failure, we shift the cell-IDs to be zero
    indexed
    """
    
    d = scipy.io.loadmat(mat_file)
    
    data = d['kn_allContactData_Interfaces_duplCorr_output_IDconv']

    FROM_COL = 0
    TO_COL = 1 

    cell_ids = np.unique(np.vstack([data[:, FROM_COL], data[:, TO_COL]])).astype(int)
    CELL_N = len(cell_ids)
    assert np.min(cell_ids) == 1
    
    # some cells have ZERO synapses, so they are empty; still nto sure where cellids
    # > 1123 come from 
    
    cellid_to_pos = {}
    for i, id in enumerate(cell_ids):
        cellid_to_pos[int(id)] = i

    # create the matrix
    area_mat = np.zeros((CELL_N, CELL_N), dtype=[('area', np.float32), 
                                                 ('count', np.uint32)])
    synapse_pos = {}

    for from_id, to_id, area, x, y, z in data:
        from_i = cellid_to_pos[int(from_id)]
        to_i = cellid_to_pos[int(to_id)]
        if area_mat[from_i, to_i] > 0:
            area_mat[from_i, to_i]['area'] += area
            area_mat[from_i, to_i]['count'] += 1

        if (int(from_id), int(to_id)) not in synapse_pos:
            synapse_pos[(int(from_id), int(to_id))] = []

        synapse_pos[(int(from_id), int(to_id))].append((x/1000., y/1000., z/1000.))
    pickle.dump({'area_mat' : area_mat, 
                 'synapse_pos' : synapse_pos, 
                 'cellid_to_pos' : cellid_to_pos, 
                 'cell_ids' : cell_ids}, 
                open(output_file, 'w'))

@files("../../../data/mouseretina/Helmstaedter_et_al_SUPPLinformation4.xlsx", 
       "xlsxdata.pickle")
def load_xlsx_data(xlsx_file, output_file):
    neuron_connect =  open_workbook(xlsx_file)
    connections = {}
    s =  neuron_connect.sheets()[0]
    N = 1123
    data = np.zeros((N, N), dtype=np.float32)
    for cell_i in range(N):
        for cell_j in range(N):
            print cell_i, cell_j
            syn_area = s.cell(cell_i + 1, cell_j+1).value
            data[cell_i, cell_j] = syn_area

    s = neuron_connect.sheets()[2]
    types = np.zeros(N, dtype=np.uint32)
    for cell in range(N):
        cell_id = s.cell(cell+1, 0).value
        type_id = s.cell(cell +1, 3).value
        types[cell] = type_id
    
    pickle.dump({'area_mat' : data, 
                 'types' : types}, 
                open(output_file, 'w'))

@files(load_data, ['synapse_hist.pdf'])
def sanity_check(infile, (synapse_hist,)):
    """
    Sanity checking and plotting
    """

    d = pickle.load(open(infile))
    area_mat = d['area_mat']
    synapse_pos = d['synapse_pos']

    print "MISSING", set(range(1, 1025)) - set(d['cell_ids'])

    f = pylab.figure(figsize=(8, 6))
    nonzero = area_mat.flatten()
    nonzero = nonzero[nonzero['count']>0]
    print nonzero['count']
    ax_count = f.add_subplot(1,2, 1)
    ax_count.hist(nonzero['count'], bins=20)
    ax_count.set_title("histogram of synapse counts")
    
    ax_total_area = f.add_subplot(1,2, 2)
    ax_total_area.hist(nonzero['area'], bins=20)
    ax_total_area.set_title("total area distribution")
    
    f.savefig(synapse_hist)


@files(load_data, "synapse_pos.png")
def plot_synapses(infile, outfile):
    """
    normalize positions to the inbound light
    """
    d = pickle.load(open(infile))
    area_mat = d['area_mat']
    synapse_pos = d['synapse_pos']

    all_synapses = []
    for k, x in synapse_pos.iteritems():
        all_synapses += x

    all_synapses = np.array(all_synapses)


    f = pylab.figure(figsize=(8, 8))

    alpha = 0.01
    s = 1.0
    for i in range(3):
        ax = f.add_subplot(2, 2, i+1)
        
        ax.scatter(all_synapses[:, (i) % 3], all_synapses[:, (i+1)%3], 
                   edgecolor='none', s=s, alpha=alpha, c='k')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

    f.savefig(outfile, dpi=600)

@files(load_data, "cell_adj.png")
def plot_adj(infile, outfile):
    """
    normalize positions to the inbound light
    """
    d = pickle.load(open(infile))
    area_mat = d['area_mat']
    CELL_N = len(area_mat)
    p = np.random.permutation(CELL_N)
    area_mat_p = area_mat[p, :]
    area_mat_p = area_mat_p[:, p]

    f = pylab.figure(figsize=(8, 8))
    ax = f.add_subplot(1, 1, 1)

    ax.imshow(area_mat_p['count'] > 0, interpolation='nearest', 
              cmap=pylab.cm.Greys)
    
    f.savefig(outfile, dpi=600)

BASEDIR = "../../../data/mouseretina"

def cell_image_files():

    for directory in [os.path.join(BASEDIR, "nature12346-s6/Supp_Info6a/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s6/Supp_Info6b/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s7/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s8/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s9/Supp_Info6e/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s9/Supp_Info6f/*.png")]:
        files = glob.glob(directory)
        for filename in files:
            f = os.path.basename(filename)
            cell_id = int(f[5:9])
            out_basename = "imgproc/%04d" % cell_id
            yield filename, [out_basename + '.png', out_basename + ".pickle"]

@files(cell_image_files)
def process_image_pos(filename, (output_png, output_pickle)):
    print "PROCESSING", filename
    W = 200
    H = 200
    R = 50
    template = np.zeros((H, W))
    a = skimage.draw.circle(H/2, W/2, R, (H, W))
    template[a] = 1
    
    REGION_X = 0, 1900
    REGION_Y = 0, 3800

    results = {}
    

    x = skimage.io.imread(filename)
    x_sub = x[REGION_Y[0]:REGION_Y[1], REGION_X[0]:REGION_X[1]]

    tgt = (x_sub[:, :, 0] < 200) & (x_sub[:, :, 1] < 200) & (x_sub[:, :, 2] < 240) & (x_sub[:, :, 2] >100)

    m = skimage.feature.match_template(tgt, template, pad_input=True)       

    tholded = m > 0.22
    label_image = skimage.morphology.label(tholded)

    coords_x_y = []
    for region in skimage.measure.regionprops(label_image, ['Area', 'Centroid']):

        # skip small images

        if region['Area'] < 1000:
            continue
        c = region['Centroid']

        coords_x_y.append((c[1], c[0]))

    if len(coords_x_y) != 4:
        print "DANGER ERROR", coords_x_y
        pylab.subplot(1, 2, 1)
        pylab.imshow(tholded)
        pylab.subplot(1, 2, 2)
        pylab.imshow(x_sub)
        pylab.show()
        raise RuntimeError("found %d coords in file %s" % (len(coords_x_y), filename))

    results = {'filename' : filename, 
               'coords' : coords_x_y}
    pickle.dump(results, open(output_pickle, 'w'))
    
    f = pylab.figure()
    ax  = f.add_subplot(1, 1, 1)
    ax.imshow(x_sub)
    for c in coords_x_y:
        ax.scatter(c[0], c[1], c='r')
    f.savefig(output_png)
    f.clf()
    del f

@merge(process_image_pos, "soma.positions.pickle")
def merge_positions(inputfiles, outputfile):
    PIX_PER_UM = 7.2
    # use the top two plots
    out_pos = {}
    out_coords = {}
    N = len(inputfiles)
    coords = np.zeros((N, 3), dtype=np.float32)
    for f in inputfiles:
        cell_id = int(os.path.basename(f[1])[:4])
        d = pickle.load(open(f[1]))

        # NOTE THESE ARE IN BS UNITS
        x = 0
        y = 0
        z = 0
        for c in d['coords']:
            if c[0] < 900 and c[1] < 900:
                # upper left plot in image, meaning x-z
                x = c[0] / PIX_PER_UM
                z = c[1] / PIX_PER_UM
            elif c[0] > 900 and c[1] < 900:
                y = c[0] / PIX_PER_UM
                z = c[1] / PIX_PER_UM
        
                
        out_coords[cell_id] = d['coords']
        out_pos[cell_id] = (x, y, z)
        coords[cell_id -1] = (x, y, z)

    pickle.dump({'pos' : out_pos, 
                 'coords_px' : out_coords, 
                 'pos_vec' : coords}, open(outputfile, 'w'))
    
@files("../../../data/mouseretina/types.xlsx",
       "type_metadata.pickle")
def type_metadata(xlsx_file, output_file):
    neuron_connect =  open_workbook(xlsx_file)

    s =  neuron_connect.sheets()[0]
    N = 71
    res = []
    
    for cell_i in range(N):
        cell_id = s.cell(cell_i + 1, 0).value
        desig = s.cell(cell_i + 1, 1).value
        volgyi = s.cell(cell_i + 1,2).value
        macneil = s.cell(cell_i + 1, 3).value
        certainty = s.cell(cell_i + 1, 4).value
        if volgyi != "":
            other = volgyi
        else:
            other=macneil
        res.append({'id' : cell_id, 
                    'desig' : desig, 
                    'other' : other, 
                    'certainty' : certainty})
    df = pandas.DataFrame(res)
    df = df.set_index(df['id'])
    del df['id']
                    
    pickle.dump({'type_metadata' : df}, 
                open(output_file, 'w'))

            
pipeline_run([load_synapse_data, load_xlsx_data, sanity_check, plot_synapses, 
              plot_adj, process_image_pos, merge_positions, type_metadata])

