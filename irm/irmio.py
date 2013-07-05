import numpy as np

import models
import util
import model
import relation
import pyirmutil

def model_from_config_file(configfile):
    config = pickle.load(open(configfile, 'r'))
    return model_from_config(config)

def model_from_config(config, relation_class=pyirmutil.Relation, 
                      rng=None):

    domains_config = config['domains']
    relations_config = config['relations']
    data_config = config['data']
    ss_config = config.get('ss', {})

    # build the model
    relations = {}
    domains_to_relations = {}
    domains_to_relations_pos = {}
    for t in domains_config:
        domains_to_relations[t] = []
        domains_to_relations_pos[t] = {}
    for rel_name, rel_config in config['relations'].iteritems():
        domaindef = [(tn, domains_config[tn]['N']) for tn in rel_config['relation']]
        if rel_config['model'] == "BetaBernoulli":
            m = models.BetaBernoulli()
        elif rel_config['model'] == "BetaBernoulliNonConj":
            m = models.BetaBernoulliNonConj()
        elif rel_config['model'] == "LogisticDistance":
            m = models.LogisticDistance()
        else:
            raise NotImplementedError()
        rel = relation_class(domaindef, data_config[rel_name], 
                             m)
        rel.set_hps(rel_config['hps'])

        relations[rel_name] = rel
        # set because we only want to add each relation once to a domain
        for tn in set(rel_config['relation']):
            domains_to_relations[tn].append((tn, rel))
            rl = len(domains_to_relations_pos[tn])
            domains_to_relations_pos[tn][rel_name] = rl

    domain_interfaces = {}
    for d_name, d_config in domains_config.iteritems():
        D_N = d_config['N'] 
        ti = model.DomainInterface(D_N, domains_to_relations[d_name])
        ti.set_hps(d_config['hps'])
        domain_interfaces[d_name] = ti

        
    irm_model = model.IRM(domain_interfaces, relations)

    domain_assignvect_to_gids = {}
    for dn, di in domain_interfaces.iteritems():
        assign_vect = domains_config[dn]['assignment']
        gr = {}
        for ai, a in enumerate(assign_vect):
            if a not in gr:
                gr[a] = di.create_group(rng)
            di.add_entity_to_group(gr[a], ai)
        domain_assignvect_to_gids[dn] = gr

    # now load / set the sufficient statistics
    for relname, reldata in ss_config.iteritems():
        rel_obj = relations[relname]
        d_names = relations_config[relname]['relation']
        domain_rel_pos = [domains_to_relations_pos[dn][relname] for dn  in d_names]
        dr = zip([domain_interfaces[dn] for dn in d_names], domain_rel_pos)

        for assignvect_group_coord, ss_val in reldata.iteritems():
            gid_group_coord = []
            for d, g in zip(d_names, assignvect_group_coord):
                gid_group_coord.append(domain_assignvect_to_gids[d][g])
            gid_group_coord = tuple(gid_group_coord)
            
            model.set_components_in_relation(dr, rel_obj, 
                                             gid_group_coord, ss_val)

    return irm_model

def empty_domain(domain_obj):
    """
    Remove all objects and delete all groups
    """
    for ei in range(domain_obj.entity_count()):
        gid = domain_obj.remove_entity_from_group(ei)
        if domain_obj.group_size(gid) == 0:
            domain_obj.delete_group(gid)
            
def init_domain(domain_obj, assign_vect):
    """
    initialize a domain to a particular assignment vector,
    returning the mapping of assignment-vector-val to
    new GID
    """
    assert domain_obj.entity_count() == len(assign_vect)
    empty_domain(domain_obj)
    ai_to_gid = {}
    for ei, ai in enumerate(assign_vect):
        if ai not in ai_to_gid:
            ai_to_gid[ai] = domain_obj.create_group(rng)
        domain_obj.add_entity_to_group(ai_to_gid[ai], ei)
    return ai_to_gid
    
def default_graph_init(connectivity, model = 'BetaBernoulli'):
    """
    Create a default IRM config from a graph connectivity matrix
    """
    T1_N = connectivity.shape[0]
    assert connectivity.shape[0] == connectivity.shape[1]
    config = {'domains' : {'d1' : {'hps' : 1.0, 
                                 'N' : T1_N}},
              'relations' : { 'R1' : {'relation' : ('d1', 'd1'), 
                                      'model' : model, 
                                      'hps' : {'alpha' : 1.0, 
                                               'beta' : 1.0}}}, 
              'data' : {'R1' : connectivity}}

    return config
