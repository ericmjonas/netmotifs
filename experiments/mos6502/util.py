import networkx as nx

def remove_merge_wires(g, ntype='wire'):
    """
    for all nodes that are 'wires', remove the node and connect
    all of their neighborhood directly to one another. 
    """
    # get the list of wire nodes
    wires = [n for n in g.nodes() if g.node[n]['ntype'] == 'wire']
    
    for node_w in wires:
        neighbors = g.neighbors(node_w)
        g.remove_node(node_w)
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 != n2:
                    g.add_edge(n1, n2)


    
