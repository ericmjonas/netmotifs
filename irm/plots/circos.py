import numpy as np 
from jinja2 import Template
import os
import subprocess
import tempfile
import shutil

SRC_DIR = os.path.dirname(__file__)
INKSCAPE_BIN = "/Applications/Inkscape.app/Contents/Resources/bin/inkscape"


def read_template(x):
    f = os.path.join(SRC_DIR, x)
    return open(f, 'r').read()
    

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class CircosPlot(object):
    """
    There are two canonical representations in our plotting layer: 
    1. Position of an entity -- controls where it is plotted intra-group
    2. group id: controls which order it is plotted in 
    """

    def __init__(self, init_assign_vect, ideogram_radius="0.8r", 
                 ideogram_thickness="40p", 
                 karyotype_colors=None, 
                 custom_color_map = None):
        self.init_assign_vect = init_assign_vect
        self.karyotype_colors = karyotype_colors

        # compute chromosomes
        self.chromosome_ids_ordered = sorted(np.unique(init_assign_vect))

        self.chromosomes = {}
        self.id_to_chrom_pos = {}
        for ci in self.chromosome_ids_ordered:
            self.chromosomes[ci] = np.argwhere(self.init_assign_vect == ci).flatten()
            for c_pos, c in enumerate(self.chromosomes[ci]):
                self.id_to_chrom_pos[c] = (ci, c_pos)

        self.labels = None
        self.labels_config = {'label_size' : '40p'}
        self.links = None
        self.class_ribbons = []
        self.plots = []

        self.ideogram_radius = ideogram_radius
        self.ideogram_thickness = ideogram_thickness
        self.custom_color_map = custom_color_map

    def set_entity_labels(self, labels, **kargs):
        assert len(labels) == len(self.init_assign_vect)
        self.labels = labels
        self.labels_config.update(**kargs)

    def set_entity_links(self, links):
        """
        entity links are from one entity to another
        # FIXME eventually let them have scalars
        [(ent1, ent2), ...]
        """
        self.links = links

    def set_class_ribbons(self, ribbons):
        self.add_class_ribbons(ribbons)

    def add_class_ribbons(self, ribbons, color='black_a5'):
        """
        list of (source group, dest group, width)
        """
        self.class_ribbons.append((ribbons, color))

    def add_plot(self, plot_type, config, data, otherconfig=None):
        """
        The assumption at the moment is that the data map 
        1:1 with the entity positions
        """
        if plot_type == 'scatter':
            default = {'glyph' : 'rectangle', 
                       'glyph_size' : 8, 
                       'color' : 'red', 
                       'stroke_color' : 'dred', 
                       'stroke_thickness' : 1}

            default.update(config)
            config = default

        elif plot_type == 'heatmap':
            default = {
                       'stroke_color' : 'black', 
                       'stroke_thickness' : 1}

            default.update(config)
            config = default

        elif plot_type == 'text':
            default = {'color' : 'black', 
                       'label_size' : '20p', 
                       'label_font' : 'condensed', 
            }

            default.update(config)
            config = default

        elif plot_type == 'line':
            default = {
                       'color' : 'black', 
                       'stroke_color' : 'black', 
                       'stroke_thickness' : 1}

            default.update(config)
            config = default


        else:
            raise Exception("I don't know this plot type %s" % plot_type)

        if otherconfig == None:
            otherconfig = {}

        self.plots.append((plot_type, config, data, otherconfig))
        
def write(config, outfilename, tempdir=None):
    """
    tempdir : where we put the intermediate outputs
    """
    
    if tempdir == None:
        tempdir = tempfile.mkdtemp()
    outfilename = os.path.abspath(outfilename)

    with cd(tempdir):
        # first write the karyotype file
        karyotype_template = Template(read_template("circos_karyotype.template"))

        # build chromosome list
        chl = []


        for chromosome_id in config.chromosome_ids_ordered:
            c = config.chromosomes[chromosome_id]
            chl.append({'name' : 'c%d' % chromosome_id, 
                       'label' : 'c%d' % chromosome_id, 
                       'entities' : c})
        karyotype_str = karyotype_template.render(chromosomes = chl, 
                                                  colors = config.karyotype_colors)
        fid = open('karyotype.txt', 'w')
        fid.write(karyotype_str)
        fid.close()

        # now the labels
        if config.labels != None:
            label_template = Template(read_template("circos_structure_label.template"))
            labels = []
            # write labels
            for id, (chromosome_id, chromosome_pos) in config.id_to_chrom_pos.iteritems():
                labels.append({'id' : id, 'chr' : "c%d" % chromosome_id, 
                               'pos' : chromosome_pos, 'label' : config.labels[id]})
            label_str = label_template.render(labels =  labels)
            fid = open("structure.label.txt", 'w')
            fid.write(label_str)
            fid.close()

        # now the links
        if config.links != None:
            links_template = Template(read_template("circos_links.template"))
            links = []
            # write linkss
            for src_id, dest_id in config.links:
                s = config.id_to_chrom_pos[src_id]
                d = config.id_to_chrom_pos[dest_id]
                links.append({'src_c' : "c%d" % s[0], 'src_i' : s[1], 
                              'dest_c' : "c%d" % d[0], 'dest_i' : d[1]})
            links_str = links_template.render(links = links)
            fid = open("links.txt", 'w')
            fid.write(links_str)
            fid.close()

        # write the ribbons
        for ribbon_i, (class_ribbons, ribbon_color) in enumerate(config.class_ribbons):
            ribbon_template = Template(read_template("circos_ribbons.template"))
            ribbons = []
            # write wribbons
            for src_id, dest_id, width in class_ribbons:
                ribbons.append({'src_c' : "c%d" % src_id, 
                                'src_len' : len(config.chromosomes[src_id]), 
                                'dest_c' : "c%d" % dest_id, 
                                'dest_len' : len(config.chromosomes[dest_id]), 
                                'width' : width})

            ribbons_str = ribbon_template.render(ribbons= ribbons)
            fid = open("ribbons.%d.txt" % ribbon_i, 'w')
            fid.write(ribbons_str)
            fid.close()
        
        # write the plots
        for plot_i, (plot_type, plot_config, plot_data, 
                     extra_config) in enumerate(config.plots):
            fid = open("plot.%d.txt" % plot_i, 'w')
            
            for entity_i, entity_val in enumerate(plot_data):
                chrom_id, chrom_pos = config.id_to_chrom_pos[entity_i]
                if np.isfinite(entity_val):
                    fid.write("c%d %d %d %s\n" % (chrom_id, chrom_pos, chrom_pos+1, entity_val))

            fid.close()
        # now write the config

        conf_template = Template(read_template("circos_conf.template"))
        conf_str = conf_template.render(has_labels = (config.labels != None), 
                                        has_links = (config.links != None), 
                                        ribbons = config.class_ribbons, 
                                        labels_config = config.labels_config, 
                                        plots = config.plots, 
                                        ideogram_radius=config.ideogram_radius, 
                                        ideogram_thickness = config.ideogram_thickness, 
                                        custom_color_map = config.custom_color_map)
        
        fid = open("circos.conf", 'w')
        fid.write(conf_str)
        fid.close()

        # invoke circos
        subprocess.call(['circos',  '-conf', "circos.conf"])
        if "png" in outfilename:
            shutil.copyfile("circos.png", outfilename)
        elif "svg" in outfilename:
            shutil.copyfile("circos.svg", outfilename)
        # elif "pdf" in outfilename: # NOTE THIS PRODUCES REALLY SHITTY PDFS
        #     subprocess.call([INKSCAPE_BIN, "-z", "-D", "--file=circos.svg", 
        #                      "--export-pdf=%s" % outfilename])
        else:
            raise Exception("unknown output file type %s" % outfilename)
        

        
