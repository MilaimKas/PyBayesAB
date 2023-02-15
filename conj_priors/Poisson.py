
import numpy as np

from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import nbinom

import helper
import plot_functions

class Bays_poisson:
    def __init__(self, prior_a=10**-4, prior_b=1) -> None:
        """
        Class for:
        - likelihood = Poisson
        - prior and posterior = Gamma

        Contains ploting and animation function
        """

        self.data = []

        self.prior_a = prior_a
        self.prior_b = prior_b

    def add_experiment(self, n_events):
        """
        add 1 interval with nbr of event

        Args:
            n_events (int): number of occurences or events for one interval
        """
        self.data.append([n_events])
    
    def add_rand_experiment(self,n,mu):
        """
        add n intervals with random nbr of events ([n_events])
        taken from a Poisson distribution with meam mu

        Args:
            n (in): number of intervals
            mu (float): mean number of occurences
        """
        n_events = poisson.rvs(mu, size=n)
        self.data.extend(n_events)
    
    def post_pred(self):
        """
        returns the posterior predictve distribution which gives the probabilty for the next observation
        p(x|X) where x = new observation and X = all data collected so far

        Returns:
            scipy.stats.nbinom: posterior predictive distribution
        """
        a,b = self.post_parameters()
        return nbinom(a,b/(1+b))
    
    def post_parameters(self):
        """
        return the parameters for the gamma posterior given the data
        """
        a = sum(self.data)+self.prior_a
        b = len(self.data)+self.prior_b

        return a,b

    def plot_tot(self, n_rvs=1000, mui=0, muf=50):
        """
        plot the posterior distribution for the total result

        Args:
            n_rvs (int, optional): number of random values for the histogram. Defaults to 1000.
            mui (float, optional): lower limit for the mean number of occurences. Defaults to 0.
            muf (float, optional): upper limit for the mean number of occurences. Defaults to 50.
        """

        mu_pts = np.linspace(mui,muf,1000)
        a = sum(self.data)+self.prior_a
        b = len(self.data)+self.prior_b

        post = gamma(a, scale=1/b) 
        
        plot_functions.plot_tot(post, n_rvs, mu_pts, xlabel="Mean number of occurence")


    def plot_exp(self, type="1D", n_pdf=1000, mui=0, muf=None):
        """
        plot "cumulative" posteriors

        Args:
            type (str, optional): one dimensional plot with x=p and y=pdf 
                                    or 2 dimensional plot with x=exp and y=p_range and z=pdf. 
                                    Defaults to "1D".
            n_pdf (int, optional): Number of points on the x axis. Defaults to 1000.
            mui (int, optional): lower limit for p. Defaults to 0.
            muf (int, optional): upper limit for p. Defaults to 1.
        """

        if muf is None:
            muf = 50

        mu_pts = np.linspace(mui,muf,n_pdf)
        data = np.array(self.data, dtype="object")
        n_exp = len(self.data)

        # cumulative posterior parameter values
        cum_alpha = np.cumsum(data)+self.prior_a
        cum_beta = np.arange(1, len(data)+1)
            
        exp = np.arange(0, n_exp)
        zip_post_para = zip(cum_alpha, cum_beta)

        post = helper.gamma_custom

        plot_functions.plot_cum_post(post, zip_post_para, exp, mu_pts, n_pdf, post_para_label="Mean number of occurence", type=type)

    def plot_anim(self, muf, mui=0, n_pdf=1000, interval=None, list_hdi=[95,90,80,60]):
        """
        Create an animation for the evolution of the posterior

        Args:
            mui (int, optional): lower limit for the Poisson mean. Defaults to 0.
            muf (int, optional): upper limit for the Poisson mean. Defaults to 1.
            n_pdf (int, optional): number of pts for the Poisson means. Defaults to 1000.
            interval (float, optional): time in ms between frames. Defaults to None.
            list_hdi (list, optional): list of hdi's to be displayed. Defaults to [95,90,80,60].

        Returns:
            pyplot.animate.Funcanimation
        """
        
        data = np.array(self.data, dtype="object")
        
        # cumulative posterior parameter values
        cum_alpha = np.cumsum(data)+self.prior_a
        cum_beta = np.arange(1, len(data)+1)

        post_para = [(ca,cb) for ca,cb in zip(cum_alpha, cum_beta)]

        gamma_ = helper.gamma_custom

        return plot_functions.plot_anim(gamma_, post_para, mui, muf, 
                model_para_label="Mean number of occurence", list_hdi=list_hdi, n_pdf=n_pdf, interval=interval)