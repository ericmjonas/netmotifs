import numpy as np

import models
import util
import irm
import relation

def model_from_config_file(configfile):
    config = pickle.load(open(configfile, 'r'))
    return model_from_config(config)

def model_from_config(config, relation_class=relation.Relation, init='allone'):

    types_config = config['types']
    relations_config = config['relations']
    data_config = config['data']

    # build the model
    relations = {}
    types_to_relations = {}
    for t in types_config:
        types_to_relations[t] = []

    for rel_name, rel_config in config['relations'].iteritems():
        typedef = [(tn, types_config[tn]['N']) for tn in rel_config['relation']]
        if rel_config['model'] == "BetaBernoulli":
            model = models.BetaBernoulli()
        elif rel_config['model'] == "BetaBernoulliNonConj":
            model = models.BetaBernoulliNonConj()
        else:
            raise NotImplementedError()
        rel = relation_class(typedef, data_config[rel_name], 
                                     model)
        rel.set_hps(rel_config['hps'])

        relations[rel_name] = rel
        # set because we only want to add each relation once to a type
        for tn in set(rel_config['relation']):
            types_to_relations[tn].append((tn, rel))
    type_interfaces = {}
    for t_name, t_config in types_config.iteritems():
        T_N = t_config['N'] 
        ti = irm.TypeInterface(T_N, types_to_relations[t_name])
        ti.set_hps(t_config['hps'])
        type_interfaces[t_name] = ti

    irm_model = irm.IRM(type_interfaces, relations)

    # now initialize all to 1
    for tn, ti in type_interfaces.iteritems():
        if init == 'allone':
            g = ti.create_group()
            for j in range(ti.entity_count()):
                ti.add_entity_to_group(g, j)
        elif init == 'singleton':
            for j in range(ti.entity_count()):
                g = ti.create_group()
                ti.add_entity_to_group(g, j)
        elif init == "crp": 
            perm = np.random.permutation(ti.entity_count())
            # FIXME this should really be CRP
            assign = np.arange(ti.entity_count()) % 8
            gr = {}
            for ai, a in enumerate(assign):
                if a not in gr:
                    gr[a] = ti.create_group()
                ti.add_entity_to_group(gr[a], ai)
                
        else:
            raise NotImplementedError()
            
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
    assert domain_obj.entity_count() == len(assign_vect)
    empty_domain(domain_obj)
    ai_to_gid = {}
    for ei, ai in enumerate(assign_vect):
        if ai not in ai_to_gid:
            ai_to_gid[ai] = domain_obj.create_group()
        domain_obj.add_entity_to_group(ai_to_gid[ai], ei)
    
    
