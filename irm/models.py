import numpy as np
from scipy.special import betaln
import util
from scipy.stats import chi2, t, norm

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

    def sample_hps(self):
        """
        draw a sample of the HPs from some prior
        """
        return {'alpha' :  np.random.gamma(2, 2), 
                'beta' : np.random.gamma(2, 2)}

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

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """

    def param_eval(self, d, ss, hps):
        """
        At distance dist, evaluate the prob of connection
        """
        p = util.logistic(d, ss['mu'], ss['lambda'])
        p = p * (hps['p_max'] - hps['p_min']) + hps['p_min']
        return p

class LogisticDistanceFixedLambda(object):
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
        return {'lambda' :  np.random.gamma(1, 1), 
                'mu_hp' : np.random.gamma(1.2, 4), 
                'p_min' : np.random.uniform(0.01, 0.1), 
                'p_scale_alpha_hp' : 1.0, 
                'p_scale_beta_hp' :1.0}

    def sample_param(self, hps):
        """
        draw a sample 
        """
        mu = np.random.exponential(hps['mu_hp'])
        p_scale = np.random.beta(hps['p_scale_alpha_hp'], 
                                 hps['p_scale_beta_hp'])
        # fix numeric precision issues
        if p_scale < 0.0001:
            p_scale = 0.0001 
        elif p_scale > 0.9999:
            p_scale = 0.9999

        
        return {'p_scale' : p_scale, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        d = np.random.exponential(hps['mu_hp'])
        p = util.logistic(d, ss['mu'], hps['lambda'])
        
        p = p * (ss['p_scale'] - hps['p_min']) + hps['p_min']
        link = np.random.rand() < p
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = link
        return x[0]

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """

    def param_eval(self, d, ss, hps):
        """
        At distance dist, evaluate the prob of connection
        """
        p = util.logistic(d, ss['mu'], hps['lambda'])
        p = p * (ss['p_scale'] - hps['p_min']) + hps['p_min']
        return p
        

class SigmoidDistance(object):
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
        p = util.sigmoid(d, ss['mu'], ss['lambda'])
        p = p * (hps['p_max'] - hps['p_min']) + hps['p_min']
        link = np.random.rand() < p
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = link
        return x[0]


class LinearDistance(object):
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
        Draw a sample of the HPs from some prior
        """
        return {'p_alpha' :  np.random.gamma(2, 2), 
                'p_beta' :  np.random.gamma(2, 2), 
                'mu_hp' : np.random.gamma(1.2, 4), 
                'p_min' : np.random.uniform(0.01, 0.1)}


    def sample_param(self, hps):
        """
        draw a sample 
        """
        mu = np.random.exponential(hps['mu_hp'])
        p = np.random.beta(hps['p_alpha'], hps['p_beta'])
        p = np.min([0.999999, p])
        print 'Sample param, p=', p, 'mu=', mu
        return {'p' : p, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        d = np.random.exponential(hps['mu_hp'])
        p = - d * ss['p'] / ss['mu']+ ss['p']
        if d > ss['mu']:
            p = hps['p_min']

        link = np.random.rand() < p
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = link
        return x[0]

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """

class GammaPoisson(object):
    """
    Just a placeholder
    """
    def data_dtype(self):
        """
        """
        return np.uint32

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
        lamb = np.random.gamma(hps['alpha'], hps['beta'])
        
        return {'lambda' : lamb}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        return np.random.poisson(ss['lambda'])

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """


class NormalDistanceFixedWidth(object):
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
        return {'p_alpha' :  np.random.gamma(2, 2), 
                'p_beta' :  np.random.gamma(2, 2), 
                'mu_hp' : np.random.gamma(1.2, 4), 
                'p_min' : np.random.uniform(0.01, 0.1), 
                'width' : np.random.gamma(2, 1)}


    def sample_param(self, hps):
        """
        draw a sample 
        """
        mu = np.random.exponential(hps['mu_hp'])
        p = np.random.beta(hps['p_alpha'], hps['p_beta'])
        
        return {'p' : p, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        d = np.random.exponential(hps['mu_hp'])
        prob =  np.exp(-0.5 * (d - ss['mu'])**2/hps['width']**2) * ss['p'] + hps['p_min']

        link = np.random.rand() < prob
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = link
        return x[0]

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """

    def param_eval(self, d, ss, hps):
        """
        At distance dist, evaluate the prob of connection
        """
        prob =  np.exp(-0.5 * (d - ss['mu'])**2/hps['width']**2) * ss['p'] + hps['p_min']

        return prob

class SquareDistanceBump(object):
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
        return {'p_alpha' :  np.random.gamma(1, 1), 
                'p_beta' :  np.random.gamma(1, 1), 
                'mu_hp' : np.random.gamma(2., 1.)/8., 
                'p_min' : np.random.uniform(0.01, 0.1), 
                'param_weight' : 0.5, 
                'param_max_distance' : 4.0}


    def sample_param(self, hps):
        """
        draw a sample 
        """
        if np.random.rand() < hps['param_weight']:
            mu = hps['param_max_distance']
        else:
            mu = np.random.exponential(hps['mu_hp'])

        p = np.random.beta(hps['p_alpha'], hps['p_beta'])
        
        return {'p' : p, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        d = hps['param_max_distance']
        while d >= hps['param_max_distance']:
            d = np.random.exponential(hps['mu_hp'])

        if d < ss['mu']:
            p = ss['p']
        else:
            p = hps['p_min']

        link = np.random.rand() < p
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = link
        return x[0]

class ExponentialDistancePoisson(object):
    """
    Just a placeholder
    """
    def data_dtype(self):
        """
        """
        return [('link',  np.int32), 
                ('distance', np.float32)]

    
    def sample_hps(self):
        """
        draw a sample of the HPs from some prior
        """
        return {'rate_scale_hp' :  np.random.gamma(1, 1), 
                'mu_hp' : np.random.gamma(1.2, 4)}


    def sample_param(self, hps):
        """
        draw a sample 
        """
        mu = np.random.exponential(hps['mu_hp'])
        rate_scale = np.random.exponential(hps['rate_scale_hp'])
        print "PARAMETERS SAMPLED FROM HPS:", mu, rate_scale
        return {'rate_scale' : rate_scale, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM mu and not from 
        suffstats. 
        """
        d = np.random.exponential(hps['mu_hp'])
        def exp_dist(x, lamb):
            return np.exp(-x / lamb)
        rate = ss['rate_scale'] * exp_dist(d, ss['mu'])

        count = np.random.poisson(rate)
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = count
        return x[0]

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """
    def param_eval(self, d, ss, hps):
        """
        At distance dist, evaluate the prob of connection
        """
        def exp_dist(x, lamb):
            return np.exp(-x / lamb)
        rate = ss['rate_scale'] * exp_dist(d, ss['mu'])
        return rate
        

class LogisticDistancePoisson(object):
    """
    Like logistic distance fixed lambda but rate_scale can be > 1.0 
    """
    def data_dtype(self):
        """
        """
        return [('link',  np.int32), 
                ('distance', np.float32)]

    
    def sample_hps(self):
        """
        draw a sample of the HPs from some prior
        """
        return {'lambda' :  np.random.gamma(1, 1), 
                'mu_hp' : np.random.gamma(1.2, 4), 
                'rate_min' : np.random.uniform(0.01, 0.1), 
                'rate_scale_hp' : np.random.uniform(0.2, 5.0)}

    def sample_param(self, hps):
        """
        draw a sample 
        """
        mu = np.random.exponential(hps['mu_hp'])
        rate_scale = np.random.exponential(hps['rate_scale_hp'])

        # fix numeric precision issues
        if rate_scale < 0.0001:
            rate_scale = 0.0001 
        
        return {'rate_scale' : rate_scale, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        d = np.random.exponential(hps['mu_hp'])
        rate = util.logistic(d, ss['mu'], hps['lambda'])
        
        rate = rate * (ss['rate_scale'] - hps['rate_min']) + hps['rate_min']
        count = np.random.poisson(rate) 
        
        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['distance'] = d
        x[0]['link'] = count
        return x[0]

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """

    def param_eval(self, d, ss, hps):
        """
        At distance dist, evaluate the prob of connection
        """
        rate = util.logistic(d, ss['mu'], hps['lambda'])
        rate = rate * (ss['rate_scale'] - hps['rate_min']) + hps['rate_min']
        return rate
        



class NormalInverseChiSq(object):
    """
    """
    def data_dtype(self):
        """
        """
        return np.float32

    
    def sample_hps(self):
        """
        draw a sample of the HPs from some prior
        """
        return {'mu' :  np.random.normal(0, 5), 
                'kappa' : np.random.gamma(1.2, 4), 
                'sigmasq' : np.random.gamma(1.2, 4), 
                'nu' : np.random.uniform(1.0, 5.0)}

    def sample_param(self, hps):
        """
        draw a sample 

        """
        sigmasq = hps['nu'] * hps['sigmasq'] / chi2.rvs(hps['nu'])
        
        std = np.sqrt(sigmasq / hps['kappa'])
        mu = np.random.normal(hps['mu'], std)

        
        return {'sigmasq' : sigmasq, 
                'mu' : mu}

    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAM P and not from 
        suffstats. 
        """
        
        mu = ss['mu']
        sigmasq = ss['sigmasq']


        return np.random.normal(mu, np.sqrt(sigmasq))

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """

    def param_eval(self, d, ss, hps):
        """
        At distance dist, evaluate the prob of connection
        """
        raise NotImplemented("No distance dependentce") 
        

class MixtureModelDistribution(object):
    """
    Just a placeholder
    """
    def data_dtype(self):
        """
        """
        return [('len',  np.int32), 
                ('points', np.float32, (1024, ))]

    
    def sample_hps(self):
        """
        draw a sample of the HPs from some prior
        """
        return {'comp_k' :  np.random.poisson(2) + 1, 
                'dir_alpha' : np.random.uniform(0.5, 1.5), 
                'var_scale' : np.random.gamma(1.0, 2)/12.}



    def sample_param(self, hps):
        """
        Sample some  parameters from the HPs
        """
        comp_k = hps['comp_k']

        return {'mu' : [np.random.uniform(0.0001, 0.9999) for _ in range(comp_k)],
                'var' : [np.random.chisquare(1.0)*hps['var_scale'] for _ in range(comp_k)],
                'pi' : np.random.dirichlet(np.ones(comp_k) * hps['dir_alpha']).tolist()}


    def sample_data(self, ss, hps):
        """
        NOTE THIS ONLY SAMPLES FROM THE PARAMS

        """
        N = np.min([1024, np.random.poisson(50) + 1])
        data = np.zeros(1024, dtype=np.float32)
        for n in range(N):
            # pick the component
            k = np.argwhere(np.random.multinomial(1, ss['pi'])).flatten()
            v = np.random.normal(ss['mu'][k], np.sqrt(ss['var'][k]))
            data[n] = v

        x = np.zeros(1, dtype=self.data_dtype())
        x[0]['len'] = N
        x[0]['points'] = data
        return x[0]

    def est_parameters(self, data, hps):
        """
        A vector of data for this component, and the hypers
        
        """
    def param_eval(self, d, ss, hps):
        """
        At distance dist, evaluate the prob of connection
        """
        # not relevant

NAMES = {'BetaBernoulli' : BetaBernoulli, 
         'BetaBernoulliNonConj' : BetaBernoulliNonConj, 
         'SigmoidDistance' : SigmoidDistance, 
         'LogisticDistance' : LogisticDistance, 
         'LogisticDistanceFixedLambda' : LogisticDistanceFixedLambda, 
         'LinearDistance' : LinearDistance, 
         'GammaPoisson' : GammaPoisson, 
         'NormalDistanceFixedWidth': NormalDistanceFixedWidth, 
         'SquareDistanceBump' : SquareDistanceBump, 
         'ExponentialDistancePoisson' : ExponentialDistancePoisson, 
         'LogisticDistancePoisson' : LogisticDistancePoisson, 
         'NormalInverseChiSq' : NormalInverseChiSq, 
         'MixtureModelDistribution' : MixtureModelDistribution, 
}

