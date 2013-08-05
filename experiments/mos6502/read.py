import simplejson as json
import pandas
import os
import networkx as nx
import numpy as np
from matplotlib import pylab
from ruffus import * 
import cPickle as pickle
import util

DATA_DIR = "../../../data/netlists"
@files(os.path.join(DATA_DIR, "data.json"), 'data.pickle')
def load_data(input_file, output_file):
    data = json.load(open(input_file, 'r'))
    nodenames = data['nodenames']
    nodei_to_name = {v:k for k, v in nodenames.items()}
    transdefs = data['transdefs']
    segdefs = data['segdefs']

    # What they call nodes we really call edges
    nodes = {}

    for seg in segdefs:
        w = seg[0]
        if w not in nodes:
            nodes[w] = {'segs': [], 
                        'pullup' : seg[1] == '+', 
                        'gates' : [],
                        'c1c2s' : []}
        nodes[w]['segs'].append(seg[3:])

    # trans are the 
    transistors = {}
    for tdef in transdefs:
        name = tdef[0]
        gate = tdef[1]
        c1 = tdef[2]
        c2 = tdef[3]
        bb = tdef[4]
        trans = {'name' : name, 
                 'on' : False, 
                 'gate' : gate, 
                 'c1' : c1, 
                 'c2' : c2, 
                 'bb' : bb}


        nodes[gate]['gates'].append(name)
        nodes[c1]['c1c2s'].append(name)
        nodes[c2]['c1c2s'].append(name)
        transistors[name] = trans

    # sort nodes by gate count

    df = pandas.DataFrame({'pullup' : [n['pullup'] for n in nodes.values()],
                           'gates' : [len(n['gates']) for n in nodes.values()], 
                           'c1c2s' : [len(n['c1c2s']) for n in nodes.values()]}, 
                          index=nodes.keys())
    df['name'] = pandas.Series(nodei_to_name)

    print "nodes sorted by gates" 
    
    result = df.sort(['gates'], ascending=False)
    print result.head(10)

    print "nodes sorted by c1c2s"
    result = df.sort(['c1c2s'], ascending=False)
    print result.head(10)

    tfdf = pandas.DataFrame(transistors.values(), index=transistors.keys())

    #tfdf['x'] = Series([tfdf['bb'][0] - tfdf['bb'][1])/2. 
    tfdf['x'] =  tfdf['bb'].map(lambda x : (x[0] + x[1])/2.0)
    tfdf['y'] =  tfdf['bb'].map(lambda x : (x[2] + x[3])/2.0)

    # f = pylab.figure()
    # ax = f.add_subplot(1, 1, 1)
    # ax.scatter(tfdf['x'], tfdf['y'], s=5, edgecolor='none', alpha=0.5)
    # f.savefig('test.pdf')

    # f2 = pylab.figure(figsize=(10, 10))
    # g=nx.Graph()
    # g.add_nodes_from(df.index, ntype='wire')
    # g.add_nodes_from(tfdf['name'], ntype='transistor')

    # for rowi, row in tfdf.iterrows():
    #     t_node = row['name']
    #     for pin in ['gate', 'c1', 'c2']:
    #         g.add_edge(t_node, row[pin], pin =pin)
    #         edgei += 1

    # print len(df.index)
    # print len(tfdf['name'])
    # print g.number_of_nodes()
    # print g.number_of_edges()
    # print edgei

    # # now delete a few massive-binding-nodes
    # nodes_to_delete = ['vss', 'vcc']
    # for n in nodes_to_delete:
    #     r = df[df['name'] == n].iloc[0]
    #     g.remove_node(r.name)
        
    # for n in g.nodes():
    #     if g.node[n]['ntype'] == 'wire':
    #         g.node[n]['fillcolor'] = '#FF000080'
    #         wire_name = df.loc[n]['name']
    #         if isinstance(wire_name, float):
    #             wire_name = ""
    #         g.node[n]['label'] = wire_name
    #     elif g.node[n]['ntype'] == 'transistor':
    #         g.node[n]['fillcolor'] = '#0000FF80'
    #     g.node[n]['style'] = 'filled'
    # nx.write_dot(g, "test.dot")

    #pos=nx.graphviz_layout(g)
    #nx.draw(g, pos=pos)
    #f2.savefig('test.graph.png', dpi=300)
    # add nodes that represent indices
    
    # add nodes that represent vertices 

    # 
    pickle.dump({'tfdf' : tfdf, 'wiredf'  : df}, 
                open(output_file, 'w'))

@files(load_data, "transistors.pdf")
def plot_transistors(input_file, output_file):
    d = pickle.load(open(input_file, 'r'))
    tfdf = d['tfdf']

    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.scatter(tfdf['x'], tfdf['y'], s=5, edgecolor='none', alpha=0.5)
    f.savefig(output_file)

@files(load_data, "graph.pickle")
def create_raw_graph(input_file, output_file):
    d = pickle.load(open(input_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    g=nx.Graph()
    g.add_nodes_from(wiredf.index, ntype='wire')
    g.add_nodes_from(tfdf['name'], ntype='transistor')

    for rowi, row in tfdf.iterrows():
        t_node = row['name']
        for pin in ['gate', 'c1', 'c2']:
            g.add_edge(t_node, row[pin], pin =pin)

    pickle.dump({'graph' : g}, 
                open(output_file, 'w'))

@files([load_data, create_raw_graph], 'rawgraph.dot')
def plot_raw_graph((data_file, graph_file), output_file):

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
    
    nodes_to_delete = ['vss', 'vcc']
    for n in nodes_to_delete:
        r = wiredf[wiredf['name'] == n].iloc[0]
        g.remove_node(r.name)
        
    for n in g.nodes():
        if g.node[n]['ntype'] == 'wire':
            g.node[n]['fillcolor'] = '#FF000080'
            wire_name = wiredf.loc[n]['name']
            if isinstance(wire_name, float):
                wire_name = ""
            g.node[n]['label'] = wire_name
        elif g.node[n]['ntype'] == 'transistor':
            g.node[n]['fillcolor'] = '#0000FF80'
        g.node[n]['style'] = 'filled'
    nx.write_dot(g, output_file)

    #sfdp -Tpdf test.dot -o test.pdf -v -Goverlap=prism

@files([load_data, create_raw_graph], 'mergedgraph.pickle')
def merge_wires_graph((data_file, graph_file), output_file):
    """
    Merge the wires into the transistor nodes
    and also add spatial data
    """

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
    
    nodes_to_delete = ['vss', 'vcc']
    for n in nodes_to_delete:
        r = wiredf[wiredf['name'] == n].iloc[0]
        g.remove_node(r.name)

    util.remove_merge_wires(g)
    print g.number_of_nodes()
        
    # for n in g.nodes():
    #     if g.node[n]['ntype'] == 'wire':
    #         g.node[n]['fillcolor'] = '#FF000080'
    #         wire_name = wiredf.loc[n]['name']
    #         if isinstance(wire_name, float):
    #             wire_name = ""
    #         g.node[n]['label'] = wire_name
    #     elif g.node[n]['ntype'] == 'transistor':
    #         g.node[n]['fillcolor'] = '#0000FF80'
    #     g.node[n]['style'] = 'filled'
    #nx.write_dot(g, output_file)
    pickle.dump({"graph" : g}, 
                open(output_file, 'w'))

    #sfdp -Tpdf test.dot -o test.pdf -v -Goverlap=prism

@files(merge_wires_graph, 'mergedgraph.dot')
def plot_merged_graph(graph_file, output_file):

    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
    nx.write_dot(g, output_file)

    #sfdp -Tpdf test.dot -o test.pdf -v -Goverlap=prism

@files([load_data, create_raw_graph],"analysis.txt")
def analysis((data_file, graph_file), out_file):

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']

    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
        
    for n in g.nodes():
        if g.node[n]['ntype'] == 'wire':
            g.node[n]['fillcolor'] = '#FF000080'
            wire_name = wiredf.loc[n]['name']
            if isinstance(wire_name, float):
                wire_name = ""
            g.node[n]['label'] = wire_name
        elif g.node[n]['ntype'] == 'transistor':
            g.node[n]['fillcolor'] = '#0000FF80'
            g.node[n]['label'] = str(n)
        else:
            raise NotImplementedError()
        g.node[n]['style'] = 'filled'
    

    vcc = wiredf[wiredf['name'] == 'vcc'].iloc[0].name
    vss = wiredf[wiredf['name'] == 'vss'].iloc[0].name

    # how many nodes have vcc on a c1 or c2
    print "c1 -> vcc: ", len(tfdf[tfdf['c1'] == vcc])
    print "c2 -> vcc: ", len(tfdf[tfdf['c2'] == vcc])
    print "gate -> vcc: ", len(tfdf[tfdf['gate'] == vcc])
    
    print "c1 -> vss: ", len(tfdf[tfdf['c1'] == vss])
    print "c2 -> vss: ", len(tfdf[tfdf['c2'] == vss])
    print "gate -> vss: ", len(tfdf[tfdf['gate'] == vss])
    
    IGNORE_WIRES = set([vcc, vss])
    # get the neighbors 
    tgt_trans = ['t%03d' % (i*100) for i in range(1, 30)]

    for t in tgt_trans:
        active_set = set([t])
        ITERS = 3
        for i in range(ITERS):
            new_set = set()
            for n in active_set:
                new_set.update(set(g.neighbors(n)))
            active_set = active_set.union(new_set)

            active_set.difference_update(IGNORE_WIRES)
        active_set.update(IGNORE_WIRES)
        print len(active_set)
        sg = g.subgraph(active_set).copy()

        f = pylab.figure(figsize=(16, 16))
        ax = f.add_subplot(1,1, 1)
        labels = {k : g.node[k]['label'] for k in sg.nodes()}


        
        for pnode in IGNORE_WIRES:
            conn_to_pnode = sg.neighbors(pnode)
            pnode_name = sg.node[pnode]['label']
            for n_i, n in enumerate(conn_to_pnode):
                pin_name = sg.edge[pnode][n]['pin']
                sg.remove_edge(pnode, n)
                new_pnode_name = "%s.%d" % (pnode_name, n_i)
                sg.add_node(new_pnode_name, ntype='wire', power=True)
                sg.add_edge(n, new_pnode_name, pin=pin_name)
                labels[new_pnode_name] = pnode_name
            sg.remove_node(pnode)
            del labels[pnode]

        edge_labels = {}
        for k1, k2 in sg.edges():
            edge_labels[(k1, k2)] = sg.edge[k1][k2]['pin'] 

        node_sizes = []
        node_colors = []
        for n in sg.nodes():
            s = 1000
            c = 'r'
            if sg.node[n]['ntype'] == 'wire':
                s = 800
                c = 'w'
                if 'power' in sg.node[n]:
                    c = 'k'
                    s = 100
            if n == t:
                c = 'b'
                s = 1000
            node_sizes.append(s)
            node_colors.append(c)

        pos = nx.spring_layout(sg, scale=4.0)

        nx.draw_networkx_nodes(sg, pos, ax=ax,
                               labels = labels, 
                               node_size = node_sizes,
                               node_color = node_colors)

        nx.draw_networkx_edges(sg, pos, ax=ax)
        nx.draw_networkx_labels(sg, pos, ax=ax,
                                labels=labels, font_size=6)
        nx.draw_networkx_edge_labels(sg, pos, ax=ax,
                                     edge_labels=edge_labels, 
                                     font_size=6)

        f.savefig("graph.%s.pdf" % t)
    
    

pipeline_run([load_data, plot_transistors, create_raw_graph, 
              plot_raw_graph, #analysis, 
              plot_merged_graph])
