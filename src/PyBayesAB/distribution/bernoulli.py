

import numpy as np

from scipy.stats import beta
from scipy.stats import bernoulli

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  


from PyBayesAB import N_SAMPLE, N_PTS

PRIOR_TYPES = {
    'Bayes-Laplace': [1, 1],
    'Jeffreys': [0.5, 0.5],
    'Neutral': [1/3, 1/3],
    'Haldane': [1e-3, 1e-3]
}

PARA_RANGE = [0, 1]

class BernoulliMixin:
    def __init__(self, prior):
        """
        Class for:
        - likelihood = Bernoulli
        - prior and posterior = Beta
        """

        self.prior = prior
        self.parameter_name = "Bernoulli probability"

    def add_rand_experiment(self, n, p, group="A"):
        hits = np.sum(bernoulli.rvs(p, size=n))
        fails = n - hits
        self.add_experiment([hits, fails], group=group)

    def make_rvs(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Beta posterior needs 2 parameters: alpha and beta")
            else:
                a,b = parameters
                a += self.prior[0]
                b += self.prior[1]
        else:
            a,b = self.post_parameters(data=data, group=group)
        return beta.rvs(a,b,size=N_sample)
    
    def make_pdf(self, parameters=None, data=None, group="A", p_pts=None, para_range=PARA_RANGE):
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Beta posterior needs 2 parameters: alpha and beta")
            else:
                a,b = parameters
                a += self.prior[0]
                b += self.prior[1]
        else:
            a,b = self.post_parameters(data=data, group=group)
        if p_pts is None:
            p_pts = np.linspace(para_range[0], para_range[1], N_PTS)
        return beta.pdf(p_pts,a,b)

    def post_parameters(self, data, group="A"):
        data = np.array(self.dataA if group == "A" else self.dataB)
        a = np.sum(data[:, 0]) + self.prior[0]
        b = np.sum(data[:, 1]) + self.prior[1]
        return a,b
    
    def make_cum_post_para(self, group="A"):
        if group == "A":
            data = self.dataA
        elif group == "B":
            data = self.dataB
        else:
            raise ValueError("Only A and B are valid groups")
        data = np.array(data, dtype="object")
        # cumulative hits and fails
        cumsum_alpha = np.cumsum(data[:,0])
        cumsum_beta = np.cumsum(data[:,1])
        return cumsum_alpha,cumsum_beta

    def post_pred(self, data=None, group="A"):
        """
        returns the probability that the next data points will be a hit

        Returns:
            float: posterior predictive for a hits
        """
        a,b = self.post_parameters(data=data, group=group)
        return a/(a+b)
    
    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        if para_range is None:
            para_range=PARA_RANGE
        # create list of rvs and pdf
        a_cum, b_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        p_pts = np.linspace(para_range[0], para_range[1], N_pts)
        for a,b in zip(a_cum, b_cum):
            rvs_data.append(self.make_rvs(parameters=[a,b], N_sample=N_sample))
            pdf_data.append(self.make_pdf(parameters=[a,b], p_pts=p_pts))
        return p_pts, rvs_data, pdf_data
    

class BaysBernoulli(BernoulliMixin, BayesianModel, PlotManager):
    def __init__(self, prior_type="Bayes-Laplace"):
        BayesianModel.__init__(self)
        BernoulliMixin.__init__(self, prior=PRIOR_TYPES[prior_type])



    