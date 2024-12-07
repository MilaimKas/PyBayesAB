
import numpy as np

from scipy.stats import gamma, expon

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  
from PyBayesAB import helper

from PyBayesAB import N_SAMPLE, N_PTS

class ExponentialMixin:

    def __init__(self) -> None:
        raise NotImplementedError()

    def add_experiment(self, pts):
        return
    
    def add_rand_experiment(self,n, mean):
        return
    
    def post_pred(self, data=None):
        return
    
    def post_parameters(self,data=None):
        return
    
    def post_distr(self,data=None):
        return
    
    def plot_tot(self, mean_lower, mean_upper, data=None, n_pdf=1000):
        return
    
    def plot_anim(self, mean_lower, mean_upper, n_pdf=1000, data=None, interval=None):
        return

class BaysExponential(ExponentialMixin, BayesianModel, PlotManager):
    def __init__(self, prior=[]):
        BayesianModel.__init__(self)
        ExponentialMixin.__init__(self,  prior=prior)