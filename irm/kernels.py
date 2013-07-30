import numpy as np

def tempered_transitions(model, rng, temps,
                         latent_get, latent_set,
                         set_temp, do_inference):

    init_state = latent_get(model)
    init_score = model.get_score()
    # Up the ladder 

    current_state = init_state
 
    scores = {} # u/d ,chain_pos, dist_pos
         
    N = len(temps)
    def get_temp(i):
        return temps[i-1]

    # up transitions
    for i in range(1, N+1):


        scores[('u', i-1, i-1)] = model.get_score()
        print "Applying kernel T^_%d" %  i, "transitioning TO state x^%d"  % i
        temp = get_temp(i)
        set_temp(model, temp)
        scores[('u', i-1, i)] = model.get_score()
        do_inference(model, rng)

    print "Going down"
    for i in range(N, 0, -1):
        print "currently in state", i

        temp = get_temp(i)

        set_temp(model, temp)

        scores[('d', i, i-1)] = model.get_score()
        print "Applying kernel Tv_%d" % i, "transitioning to state Xv%d" % (i-1)
        do_inference(model, rng)
        scores[('d', i-1, i-1)] = model.get_score()
        set_temp(model, get_temp(i))
        scores[('d', i-1, i)] = model.get_score()


    score = 0
    print "up temp scores"
    for i in range(N):
        score += scores[('u', i, i+1)]  - scores[('u', i, i)]

    print "down temp scores"
    for i in range(N-1, 0, -1):
        print "i=", i
        score += scores[('d', i, i)]  - scores[('d', i, i+1)]
                                            
    if np.random.rand() < np.exp(score) :
        print "mh: accept!", score
    else:
        print "mh: reject!", score
        latent_set(model, init_state)


    
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
    
    if np.random.rand() < np.exp(s1 + s2):
        print "accept"
        ci_l = out_latents[ci]
        ci_p1_l = out_latents[ci+1]
        out_latents[ci] = ci_p1_l
        out_latents[ci+1] = ci_l
    else:
        print "reject!" 
    return out_latents
    
    
    
    
    

