import numpy as np
import gridgibbshps
import irmio
import util
import pyirmutil

def tempered_transitions(model, rng, temps,
                         latent_get, latent_set,
                         set_temp, do_inference):

    init_state = latent_get(model)
    init_score = model.get_score()
    # Up the ladder 

    current_state = init_state
 
    
    scores = {} # u/d ,chain_pos, dist_pos
    
    def set_scores(up_or_down, chain_pos, distribution_pos, score):
        k = (up_or_down, chain_pos, distribution_pos)
        if k in scores:
            raise Exception("Key %s already in scores" % str(k))

        scores[k] = score


    N = len(temps)
    def get_temp(i):
        return temps[i-1]

    set_temp(model, temps[0])

    set_scores('u', 0, 0, model.get_score())

    # up transitions
    for i in range(1, N+1):


        temp = get_temp(i)
        set_temp(model, temp)
        set_scores('u', i-1, i, model.get_score())

        print "Applying kernel T^_%d" %  i, "transitioning TO state x^%d"  % i, "temp=", temp

        set_scores('u', i, i, model.get_score())
        do_inference(model, rng, False)

    print "Going down"
    for i in range(N, 0, -1):
        print "currently in state", i

        temp = get_temp(i)

        set_temp(model, temp)

        #set_scores('d', i, i-1, model.get_score())
        print "Applying kernel Tv_%d" % i, "transitioning to state Xv%d" % (i-1), "temp=", temp
        do_inference(model, rng, True)
        print "After inference, in state", i-1
        set_temp(model, get_temp(i-1))
        set_scores('d', i-1, i-1, model.get_score())
        set_temp(model, get_temp(i))
        set_scores('d', i-1, i, model.get_score())


    score = 0
    print "up temp scores"
    for i in range(N):

        delta = scores[('u', i, i+1)]  - scores[('u', i, i)]
        print "computing p_%d(x^%d) / p_%d(x^%d)" % (i+1, i, i, i), delta
        score += delta

    print "down temp scores"
    for i in range(N-1, -1, -1):
        delta =  scores[('d', i, i)]  - scores[('d', i, i+1)]
        print "computing p_%d(Xv%d) / p_%d(Xv%d)" % (i, i, i+1, i), delta
        score += delta
                             
    p = np.exp(score)
    if np.random.rand() < p:
        print "TT: accept!", p
    else:
        print "TT: reject!", p
        latent_set(model, init_state)

    set_temp(model, temps[0])

    
def parallel_tempering(model, chain_states, 
                       rng, temps, 
                       latent_get, latent_set,
                       set_temp, do_inference, ITERS=1):
    """ 
    """
    # advance each chain ITERS states
    out_latents = []
    for cs, t in zip(chain_states, temps):
        latent_set(model, cs)
        set_temp(model, t)
        for i in range(ITERS):
            do_inference(model, rng)
        out_latents.append(latent_get(model))
        
    # propose a swap
    ci = np.random.randint(0, len(temps)-1)
    # propose a swap between ci and ci+1
    latent_set(model, out_latents[ci+1])
    set_temp(model, temps[ci])
    s1 = model.get_score()
    latent_set(model, out_latents[ci])
    s1 -= model.get_score()

    # propose a swap between ci and ci+1
    latent_set(model, out_latents[ci])
    set_temp(model, temps[ci+1])
    s2 = model.get_score()
    latent_set(model, out_latents[ci+1])
    s2 -= model.get_score()
    
    set_temp(model, temps[0])
    
    trans_str = "%d(%f) <-> %d(%f)" %(ci, temps[ci], ci+1, temps[ci+1])

    if np.random.rand() < np.exp(s1 + s2):
        print "PT accept " + trans_str
        ci_l = out_latents[ci]
        ci_p1_l = out_latents[ci+1]
        out_latents[ci] = ci_p1_l
        out_latents[ci+1] = ci_l
    else:
        print "PT reject!" + trans_str
    return out_latents
    
    
    
    
    

def anneal(model, rng, anneal_config, 
           iteration,
           set_temp, do_inference):
    """
    Simple annealing schedule. 

    """

    start_temp = anneal_config['start_temp']
    stop_temp = anneal_config['stop_temp']
    iterations = anneal_config['iterations']
    
    temps = np.logspace(np.log(start_temp), np.log(stop_temp), 
                        iterations, base=np.e)
    if iteration >= iterations:
        cur_temp = stop_temp
    else:
        cur_temp = temps[iteration]
    print "Annealing at temp=", cur_temp

    set_temp(model, cur_temp)

    res = do_inference(model, rng)
    set_temp(model, 1.0)
    return res

def domain_hp_grid(model, rng, grid):
    for domain_name, domain in model.domains.iteritems():
        
        def set_func(val):
            domain.set_hps({'alpha' : val})

        def get_score():
            return domain.get_prior_score()

        gridgibbshps.grid_gibbs(set_func, get_score, grid)

def relation_hp_grid(model, rng, grids, threadpool=None):
    """ add the ability to have per-relation grids 

    If the grid is 'None', don't do inference 

    """

    for relation_name, relation in model.relations.iteritems():
        model_name = relation.modeltypestr
        if relation_name in grids:
            grid = grids[relation_name]
        elif model_name in grids:
            grid = grids[model_name]
        else:
            raise RuntimeError("model %s is not in the provided grids" % model_name)


        if isinstance(relation, pyirmutil.Relation):
            ## THIS IS A TOTAL HACK we should not be dispatching this way
            ## fix in later version once we obsolte old code
            def set_func(val):
                relation.set_hps(val)

            def get_score():
                return relation.total_score()
            if grid == None:
                continue

            gridgibbshps.grid_gibbs(set_func, get_score, grid)
        else:
            scores = relation.score_at_hps(grid, threadpool)
            i = util.sample_from_scores(scores)
            relation.set_hps(grid[i])

def sequential_init(model, rng, M=10):
    """
    This is a sequential gibbs-style initialization. We require a model
    to be fully specified before we do this. Note that we obliterate
    all existing structural state -- components, suffstats, etc. 

    To handle the multidomain case, we randomly switch between domains
    as we do the sequential build-up. 

    Note we do neal-algo-8-style creation of ephemeral groups here

    """
    for domain_name, domain_obj in model.domains.iteritems():
        irmio.empty_domain(domain_obj)
    d_o_map = {}
    # develop ordering 
    for domain_name, domain_obj in model.domains.iteritems():
        unassigned_objs = np.random.permutation(domain_obj.entity_count()).tolist()
        d_o_map[domain_name] = unassigned_objs
    
    # now create a single group for everyone
    for domain_name, domain_obj in model.domains.iteritems():
        g = domain_obj.create_group(rng)
        domain_obj.add_entity_to_group(g, d_o_map[domain_name].pop())

    # now each domain has exactly one currently-assigned group
    
    # flatten the ordering, shuffle
    all_ent = []
    for dn, do in d_o_map.iteritems():
        all_ent += [(dn, di) for di in do]
    np.random.shuffle(all_ent)
    
    for domain_name, entity_pos in all_ent:
        domain_obj = model.domains[domain_name]

        extra_groups = [domain_obj.create_group(rng) for _ in range(M)]

        groups = domain_obj.get_groups()
        scores = np.zeros(len(groups))
        
        for gi, group_id in enumerate(groups):
            scores[gi] = domain_obj.post_pred(group_id, entity_pos)
            # correct the score for the empty groups
            if group_id in extra_groups:
                scores[gi] -= np.log(M)
        #print entity_pos, scores
        sample_i = util.sample_from_scores(scores)
        new_group = groups[sample_i]

        domain_obj.add_entity_to_group(new_group, entity_pos)
        for eg in extra_groups:
            if domain_obj.group_size(eg) == 0:
                domain_obj.delete_group(eg)
    
    # now the model init should ... be good
    # debug
    for domain_name, domain_obj in model.domains.iteritems():
        print domain_name, "groups:", 
        i = 0
        for g in domain_obj.get_groups():
            j = domain_obj.group_size(g)
            print j, 
            i += j
        print
        print "total entities", i

    

        

    

    

