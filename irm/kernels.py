import numpy as np

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
    
    
    
    
    

