import numpy as np

class BetaBernoulli(object):
    def create_hps(self):
        """
        Return a hypers obj
        """

    def create_ss(self):
        """
        """

    def data_dtype(self):
        """
        """
    def ss_add(self, ss, hp, datum):
        """
        returns updated sufficient statistics
        """
    def ss_rem(self, ss, hp, datum):
        """
        """
    def ss_score(self, ss, hp):
        pass

    def post_pred(self, ss, hp, datum):
        pass

class AccumModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self):
        return {'sum' : 0}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = ss['sum'] + datum
        ss_new = self.create_ss()
        ss_new['sum'] = s
        return ss_new
    
    def ss_rem(self, ss, hp, datum):
        s = ss['sum'] - datum
        ss_new = self.create_ss()
        ss_new['sum'] = s
        return ss_new

    def ss_score(self, ss, hp):
        return ss['sum'] + hp['offset']

    def post_pred(self, ss, hp, datum):
        return datum

class VarModel(object):
    def create_hps(self):
        return {'offset': 0.0}

    def create_ss(self):
        """
        We could make this incremental if we wanted to
        """
        return {'dp' : []}

    def data_dtype(self):
        return np.float32

    def ss_add(self, ss, hp, datum):

        s = list(ss['dp'])
        s.append(datum)
        ss_new = self.create_ss()
        ss_new['dp'] = s 
        return ss_new
    

    def ss_rem(self, ss, hp, datum):
        s = list(ss['dp'])
        s.remove(datum)
        ss_new = self.create_ss()
        ss_new['dp'] = s
        return ss_new

    def ss_score(self, ss, hp):
        s = np.var(ss['dp']) + np.mean(ss['dp']) + hp['offset']
        print ss['dp'], s

        return s

    def post_pred(self, ss, hp, datum):
        s = list(ss['dp'])
        s.append(datum)
        
        return np.var(ss) - np.var(ss['dp']) + np.mean(ss) - np.mean(ss['dp'])

