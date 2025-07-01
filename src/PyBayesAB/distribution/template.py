
import numpy as np

from scipy.stats import dirichlet
from scipy.stats import multinomial

from PyBayesAB import N_SAMPLE, N_PTS

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  


class DistMixin:
    def __init__(self, prior=None):
        return NotImplementedError
        
    def make_default_range(self, mu=None, alpha=None, beta=None, kappa=None, var="mu"):
        return NotImplementedError

    def add_rand_experiment(self, n, mu, sig, group="A"):
            return NotImplementedError

    def post_pred(self, data=None):
            return NotImplementedError

    def post_parameters(self, group="A", data=None):
        return NotImplementedError

    def get_parameters(self, group,  parameters=None, data=None):
        return NotImplementedError

    def make_rvs(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE, var="mu"):
        return NotImplementedError

    def make_pdf(self, parameters=None, data=None, group="A", p_pts=None, para_range=None, var="mu"):
        return NotImplementedError

    def make_cum_post_para(self, group="A"):
        return NotImplementedError

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, var="mu", N_pts=N_PTS):
        return NotImplementedError

    def _get_pts_range(self, p_pts, para_range, var, N_pts=N_PTS):
            return NotImplementedError
    
class BaysDist(DistMixin, BayesianModel, PlotManager):
    def __init__(self, prior=None):
        BayesianModel.__init__(self)
        DistMixin.__init__(self, prior=prior)