import numpy as np
from scipy.special import betaln
import util

class BetaBernoulli(object):
    def create_hps(self):
        """
        Return a hypers obj
        """
        return {'alpha' : 1.0, 'beta': 1.0}

    def sample_hps(self):
        """
        draw a sample of the HPs from some prior
        """
        return {'alpha' :  np.random.gamma(1, 2), 
                'beta' : np.random.gamma(1, 2)}

    def sample_param(self, hps):
        """
        draw a sample 
        """
        return {'p' : np.random.beta(hps['alpha'], 
                                     hps['beta'])}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        p = ss['p']
        return np.random.rand() < p 

    def create_ss(self, hps):
        """
        """
        return {'heads' : 0, 'tails' : 0}

    def data_dtype(self):
        """
        """
        return np.bool

    def ss_add(self, ss, hp, datum):
        """
        returns updated sufficient statistics
        """
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] +=1
        else:
            new_ss['tails'] +=1 

        return new_ss


    def ss_rem(self, ss, hp, datum):
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] -=1
        else:
            new_ss['tails'] -=1 

        return new_ss

    def ss_score(self, ss, hp):
        heads = ss['heads']
        tails = ss['tails']
        alpha = hp['alpha']
        beta = hp['beta']
        logbeta_a_b = betaln(alpha, beta)

        return betaln(alpha+heads,beta+tails) - logbeta_a_b; 

    def post_pred(self, ss, hp, datum):
        heads = ss['heads']
        tails = ss['tails']
        alpha = hp['alpha']
        beta = hp['beta']
        den = np.log(alpha+beta + heads + tails)

        if datum:
            return np.log(heads + alpha) - den
        else:
            return np.log(tails + beta) - den

class AccumModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self, hps):
        return {'sum' : 0, 
                'count' : 0}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = ss['sum'] + datum
        ss_new = self.create_ss(hp)
        ss_new['sum'] = s
        ss_new['count'] = ss['count'] + 1
        return ss_new
    
    def ss_rem(self, ss, hp, datum):
        s = ss['sum'] - datum
        ss_new = self.create_ss(hp)
        ss_new['sum'] = s
        ss_new['count'] = ss['count'] - 1
        assert ss_new['count'] >= 0

        return ss_new

    def ss_score(self, ss, hp):
        return ss['sum'] + hp['offset']

    def post_pred(self, ss, hp, datum):
        return datum

class VarModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self, hps):
        """
        We could make this incremental if we wanted to
        """
        return {'dp' : []}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = list(ss['dp'])
        s.append(datum)
        ss_new = self.create_ss(hp)
        ss_new['dp'] = s 
        return ss_new
    

    def ss_rem(self, ss, hp, datum):
        s = list(ss['dp'])
        s.remove(datum)
        ss_new = self.create_ss(hp)
        ss_new['dp'] = s
        return ss_new

    def ss_score(self, ss, hp):
        s = np.var(ss['dp']) + np.mean(ss['dp']) + hp['offset']

        return s

    def post_pred(self, ss, hp, datum):
        s = list(ss['dp'])
        s.append(datum)
        print "post_pred", s, ss, hp, datum
        return np.var(s) - np.var(ss['dp']) + np.mean(s) - np.mean(ss['dp'])

class NegVarModel(VarModel):
    def ss_score(self, ss, hp):
        s = np.var(ss['dp']) +  hp['offset']

        return -s

    def post_pred(self, ss, hp, datum):
        s = list(ss['dp'])
        s.append(datum)
        
        return -(np.var(ss) - np.var(ss['dp']))


class BetaBernoulliNonConj(object):
    """
    A non-conjugate instantaition of the beta-bernoulli conjugate
    model where we explicitly represent the P

    FIXME: Am I only keeping the sufficient statistics for MPD? 
    """
    def create_hps(self):
        """
        Return a hypers obj
        """
        return {'alpha' : 1.0, 'beta': 1.0}

    def sample_from_prior(self, hps):
        return np.random.beta(hps['alpha'], hps['beta'])

    def create_ss(self, hps):
        """
        """
        p = self.sample_from_prior(hps)

        return {'p' : p, 
                'heads' : 0, 'tails' : 0}

    def data_dtype(self):
        """
        """
        return np.bool

    def ss_add(self, ss, hp, datum):
        """
        returns updated sufficient statistics
        """
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] +=1
        else:
            new_ss['tails'] +=1 

        return new_ss


    def ss_rem(self, ss, hp, datum):
        new_ss = dict(ss)

        if datum:
            new_ss['heads'] -=1
        else:
            new_ss['tails'] -=1 

        return new_ss

    def ss_score(self, ss, hp):
        """
        This should include all sorts of stuff right? 
        """
        
        heads = ss['heads']
        tails = ss['tails']
        p = ss['p']
        alpha = hp['alpha']
        beta = hp['beta']
        logbeta_a_b = betaln(alpha, beta)
        lp = np.log(p)
        lmop = np.log(1-p)
        p_score = -logbeta_a_b + (alpha-1)*lp + (beta-1)*lmop
        
        # should this be a beta or a bernoulli
        #d_score = util.log_bernoulli(heads, tails, p)

        return p_score  # + d_score

    def post_pred(self, ss, hp, datum):
        p = ss['p']
        if datum:
            return np.log(p)
        else:
            return np.log(1-p)

class LogisticDistance(object):
    """
    Just a placeholder
    """
    def data_dtype(self):
        """
        """
        return [('link',  np.bool), 
                ('distance', np.float32)]

    
    def sample_hps(self):
        """
        draw a sample of the HPs from some prior
        """
        return {'lambda_hp' :  np.random.gamma(1, 1), 
                'mu_hp' : np.random.gamma(1.2, 4), 
                'p_min' : np.random.uniform(0.01, 0.1), 
                'p_max' : np.random.uniform(0.9, 0.99)}

    def sample_param(self, hps):
        """
        draw a sample 
        """
        lamb = np.random.exponential(hps['lambda_hp'])
        mu = np.random.exponential(hps['mu_hp'])
        
        return {'lambda' : lamb, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        d = np.random.exponential(hps['mu_hp'])
        p = util.logistic(d, ss['mu'], ss['lambda'])
        p = p * (hps['p_max'] - hps['p_min']) + hps['p_min']
        link = np.random.rand() < p
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = link
        return x[0]


class SigmoidDistance(object):
    """
    Just a placeholder
    """
    pass

NAMES = {'BetaBernoulli' : BetaBernoulli, 
         'LogisticDistance' : LogisticDistance}
