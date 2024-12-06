
import numpy as np

from scipy.stats import dirichlet
from scipy.stats import multinomial
import scipy.interpolate as interpolate

import matplotlib.pyplot as plt
from matplotlib import animation

from PyBayesAB import helper
from PyBayesAB import plot_functions

class BaysMultinomial:
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