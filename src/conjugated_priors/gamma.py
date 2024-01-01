
import numpy as np

from scipy.stats import norm, t
import scipy.interpolate as interpolate

import matplotlib.pyplot as plt
from matplotlib import animation

import src.helper as helper
import src.plot_functions as plot_functions

class BaysGammaKnownShape:
    def __init__(self) -> None:
        raise NotImplementedError()

    def add_experiment(self, pts):
        return
    
    def add_rand_experiment(self,n,sig):
        return
    
    def post_pred(self, data=None):
        return
    
    def post_parameters(self,data=None):
        return
    
    def post_distr(self,data=None):
        return
    
    def plot_tot(self, sig_lower, sig_upper, data=None, n_pdf=1000):
        return
    
    def plot_anim(self, sig_lower, sig_upper, n_pdf=1000, data=None, interval=None):
        return


class BaysGammaKnownRate:
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
    

class BaysGamma:
    def __init__(self, mu_prior=0, kappa_prior=1, alpha_prior=0.5, beta_prior=50):
        raise NotImplementedError()
    
    def add_experiment(self, pts):
        return
    
    def add_rand_experiment(self,n,mu,sig):
        return

    def post_pred(self, data=None):
        return
    
    def post_parameters(self,data=None):
        return
    
    def post_distr(self,data=None):
        return

    def plot_tot(self, mu_lower, mu_upper, sig_lower, sig_upper, data=None, n_pdf=1000):
        return
    
    def plot_anim(self, mu_lower, mu_upper, sig_lower, sig_upper, n_pdf=1000, data=None, interval=None):
        return