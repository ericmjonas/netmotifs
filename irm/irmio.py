import numpy as np

import models
import util
import model
import relation
import pyirmutil
import irm

def model_from_config_file(configfile):
    config = pickle.load(open(configfile, 'r'))
    return model_from_config(config)


def create_model_from_data(data, relation_class=pyirmutil.Relation, 
                      rng=None):

    domains_config = data['domains']
    relations_config = data['relations']


    # build the model
    relations = {}
    domains_to_relations = {}
    domains_to_relations_pos = {}
    for t in domains_config:
        domains_to_relations[t] = {}
        #domains_to_relations_pos[t] = {}
    for rel_name, rel_config in relations_config.iteritems():
        domaindef = [(tn, domains_config[tn]['N']) for tn in rel_config['relation']]
        if rel_config['model'] == "BetaBernoulli":
            m = models.BetaBernoulli()
        elif rel_config['model'] == "GammaPoisson":
            m = models.GammaPoisson()
        elif rel_config['model'] == "BetaBernoulliNonConj":
            m = models.BetaBernoulliNonConj()
        elif rel_config['model'] == "LogisticDistance":
            m = models.LogisticDistance()
        elif rel_config['model'] == "SigmoidDistance":
            m = models.SigmoidDistance()
        elif rel_config['model'] == "LinearDistance":
            m = models.LinearDistance()
        elif rel_config['model'] == "NormalDistanceFixedWidth":
            m = models.NormalDistanceFixedWidth()
        else:
            raise NotImplementedError()
        rel = relation_class(domaindef, relations_config[rel_name]['data'], 
                             m)
        relations[rel_name] = rel
        # set because we only want to add each relation once to a domain
        for tn in set(rel_config['relation']):
            domains_to_relations[tn][rel_name] = (tn, rel)
            #rl = len(domains_to_relations_pos[tn])
            #domains_to_relations_pos[tn][rel_name] = rl

    domain_interfaces = {}
    for d_name, d_config in domains_config.iteritems():
        D_N = d_config['N'] 
        ti = model.DomainInterface(D_N, domains_to_relations[d_name])
        domain_interfaces[d_name] = ti

        
    irm_model = model.IRM(domain_interfaces, relations)

    return irm_model

def set_model_latent(irm_model, latent,                       
                     rng):
    """ 
    Take in an existing model with the correct structure and types, and 
    set the latent parameters
    """

    domains_latent = latent['domains']
    relations_latent = latent['relations']

    for rel_name, relation_object in irm_model.relations.iteritems():

        relation_object.set_hps(relations_latent[rel_name]['hps'])


    for d_name, d_obj in irm_model.domains.iteritems():
        d_obj.set_hps(domains_latent[d_name]['hps'])

    domain_assignvect_to_gids = {}
    for dn, di in irm_model.domains.iteritems():
        # clear out the domain
        for i, ai in enumerate(di.get_assignments()):
            if ai != irm.model.NOT_ASSIGNED:
                di.remove_entity_from_group(i)
        gs = di.get_groups()
        for g in gs:
            di.delete_group(g)
            
        # now do the assignment
        
        assign_vect = domains_latent[dn]['assignment']
        gr = {}
        for ai, a in enumerate(assign_vect):
            if a not in gr:
                gr[a] = di.create_group(rng)
            di.add_entity_to_group(gr[a], ai)
        domain_assignvect_to_gids[dn] = gr

    # now load / set the sufficient statistics
    for relname, rel_obj in irm_model.relations.iteritems():
        d_names = rel_obj.get_axes()
        domain_rel_pos = [irm_model.domains[dn].get_relation_pos(relname) for dn  in d_names]
        dr = zip([irm_model.domains[dn] for dn in d_names], domain_rel_pos)
        rel_data = relations_latent[relname].get('ss', {})
        for assignvect_group_coord, ss_val in rel_data.iteritems():
            gid_group_coord = []
            for d, g in zip(d_names, assignvect_group_coord):
                gid_group_coord.append(domain_assignvect_to_gids[d][g])
            gid_group_coord = tuple(gid_group_coord)
            
            model.set_components_in_relation(dr, rel_obj, 
                                             gid_group_coord, ss_val)

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
    
def default_graph_init(connectivity, model = 'BetaBernoulli', extra_conn = None):
    """
    Create a default IRM config from a graph connectivity matrix. 

    Extra_conn are extra additional relations we glue on top

    """
    T1_N = connectivity.shape[0]
    assert connectivity.shape[0] == connectivity.shape[1]

    latent = {'domains' : {'d1' : {'hps' : {'alpha' : 1.0},
                                   'assignment' : np.arange(T1_N) % 50} 
                           },
              'relations' : { 'R1' : {'hps' : {'alpha' : 1.0, 
                                               'beta' : 1.0}}}}

    data = {'domains' : {'d1' : {  'N' : T1_N}, }, 
            'relations' : { 'R1' : {'relation' : ('d1', 'd1'), 
                                    'model' : model, 
                                    'data' : connectivity}}}
    if extra_conn != None:
        for c_i, connectivity in enumerate(extra_conn):
            assert T1_N ==  connectivity.shape[0]
            assert connectivity.shape[0] == connectivity.shape[1]
            r_name = 'R%d' % (c_i +2)
            latent['relations'][r_name] = {'hps' : {'alpha' : 1.0, 
                                                    'beta' : 1.0}}
            data['relations'][r_name] = {'relation' : ('d1', 'd1'), 
                                         'model' : model, 
                                         'data' : connectivity}
            

    return latent, data

def get_latent(model_obj, include_ss=True):
    domains_out = {}
    for domain_name, domain_obj in model_obj.domains.iteritems():
        hps = domain_obj.get_hps()
        a = domain_obj.get_assignments()
        N = len(a)
        domain = {'hps' : hps, 
                  'assignment' : a.tolist()}
        domains_out[domain_name] = domain
    relations_out = {}

    dom_names = model_obj.domains.keys()

    for relation_name, rel_obj in model_obj.relations.iteritems():
        model_str = rel_obj.modeltypestr
        # relation def

        hps = rel_obj.get_hps()
        relations_out[relation_name] = {'hps' : hps}
                                        
        
        doms = [(model_obj.domains[dn], model_obj.domains[dn].get_relation_pos(relation_name)) for dn in rel_obj.get_axes()] # dom_names]
        if include_ss:
            relations_out[relation_name]['ss'] = model.get_components_in_relation(doms, rel_obj)

    return {'domains' : domains_out, 
            'relations' : relations_out}

def delta_thold(x, y, tol = 0.0001):
    "Is the delta greater than the threshold?"

    if np.abs(x -y) > tol:
        return True
    else:
        return False

def latent_equality(l1, l2, data1 = None, 
                    include_ss=True, tol = 0.0001):
    domains1= l1['domains']
    domains2 = l2['domains']
    if set(domains1.keys()) != set(domains2.keys()):
        return False

    domain_groupid_maps = {}
    for d in domains1.keys():
        d1 = domains1[d]
        d2 = domains2[d]
        for hp in d1['hps'].keys():
            if delta_thold(d1['hps'][hp], d2['hps'][hp]):
                return False
        if (util.canonicalize_assignment(d1['assignment']) != util.canonicalize_assignment(d2['assignment'])).all():
            return False

        gid_map = {}
        
        domain_groupid_maps[d] = {k: v for k, v in zip(d1['assignment'], d2['assignment'])}
    relations1 = l1['relations']
    relations2 = l2['relations']
    for r in relations1.keys():
        r1 = relations1[r]
        r2 = relations2[r]
        for hp in r1['hps'].keys():
            if delta_thold(r1['hps'][hp], r2['hps'][hp]):
                return False
    if not include_ss:
        return True

    # god this is going to be a bitch
    for r in relations1.keys():
        rdef = data1['relations'][r]['relation']
        print "rdef=", rdef
        ss1 = relations1[r]['ss']
        ss2 = relations2[r]['ss']
        for g1 in ss1.keys():
            comp1 = ss1[g1]
            g2 = tuple([domain_groupid_maps[rn][g] for rn, g in zip(rdef, g1)])
            print "g2=", g2, "g1=", g1
            comp2 = ss2[g2]
            for param in comp1.keys():
                if delta_thold(comp1[param], comp2[param]):
                    return False
    return True

def estimate_suffstats(irm_model, rng, ITERS=10):
    """
    Take in a model and just perform hyperparam inference, that is, 
    don't do any structural inference. 

    This is to "fix up" the horrible init-from-prior state we 
    find ourselves in when we create a model
    """

    for i in range(ITERS):
        for relation_name, relation in irm_model.relations.iteritems():
            if relation.modeltypestr == "LogisticDistance":
                params = {'width' : relation.get_hps()['mu_hp'] / 2.0}
                relation.apply_comp_kernel("slice_sample", rng, params)
 
            elif relation.modeltypestr == "SigmoidDistance":
                params = {'width' : relation.get_hps()['mu_hp'] / 2.0}
                relation.apply_comp_kernel("slice_sample", rng, params)
 
            elif relation.modeltypestr == "LinearDistance":
                params = {'width' : relation.get_hps()['mu_hp'] / 2.0}
                relation.apply_comp_kernel("slice_sample", rng, params)

            elif relation.modeltypestr == "NormalDistanceFixedWidth":
                params = {'width' : relation.get_hps()['mu_hp'] / 2.0}
                relation.apply_comp_kernel("slice_sample", rng, params)

            elif relation.modeltypestr == "BetaBernoulliNonConj":

                relation.apply_comp_kernel("slice_sample", rng, params)

            elif relation.modeltypestr == "BetaBernoulli":
                pass
            elif relation.modeltypestr == "GammaPoisson":
                pass

            else:
                raise NotImplementedError() 

