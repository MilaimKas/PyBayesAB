

import numpy as np

from scipy.stats import beta
from scipy.stats import bernoulli

import matplotlib.pyplot as plt

import helper
import plot_functions


class Bays_bernoulli:
    def __init__(self, prior_type='Bayes-Laplace'):
        '''
        Class for:
        - likelihood = Bernoulli
        - prior and posterior = Beta

        Contains ploting and animation function
        '''

        # Define prior
        # ----------------------------------------

        # uniform Bayes-Laplace
        if prior_type == 'Bayes-Laplace':
            self.prior = [1,1]
        # Jeffreys prior
        elif prior_type == 'Jeffreys':
            self.prior = [1/2,1/2]
        # "Neutral" prior proposed by Kerman (2011)
        elif prior_type == 'Neutral':
            self.prior = [1/3,1/3]
        # Haldane prior (𝛼=𝛽=0)
        elif prior_type == 'Haldane':
            self.prior = [10**-3,10**-3]
        else:
            raise SyntaxError("Prior type not recognised")

        
        # Define liste of data
        # ------------------------------------------

        # data.shape = Nx2 where N = number of "experiments" 
        # and each experiment is define by a number of "hits" and "fails"
        # example: data = [[2,5],[1,1],[0,4]] 
        #   -> 1st experiment: 2 hits and 5 fails
        #   -> 2nd experiment: 1 hit and 1 fail
        #   -> 3nd experiment: 0 hit and 4 fails
        #   -> total result: 3 hits and 10 fails

        # group A
        self.data = []
        self.data.append(self.prior)

        # group B (for AB test)
        self.data2 = []
        self.data2.append(self.prior)
    
    def add_experiment(self, hits, fails):
        """
        add an experiment ([N_hits, N_fails]) to the data

        Args:
            hits (int): number of success
            fails (int): number of failures
        """
        self.data.append([hits,fails])
    
    def add_rand_experiment(self,n,p):
        """
        add an experiment with random trials ([N_hits, N_fails])
        taken from a Bernoulli distribution with probability p

        Args:
            n (int): number of trilas
            p (float): parameter for the Bernoulli distribution
        """
        hits = np.count_nonzero(bernoulli.rvs(p, size=n))
        self.data.append([hits,n-hits])
    
    def post_pred(self):
        """
        returns the probabilty that the next data points will be a hit

        Returns:
            float: posterior predictive for a hits
        """
        a,b = self.post_parameters()
        return a/(a+b)
    
    def post_parameters(self):
        """
        return the parameter of the Beta posterior alpha and beta given the data

        Returns:
            tuple: alpha and beta value for the Beta posterior
        """

        data = np.array( self.data, dtype="object")
        a = np.sum(data[:,0])
        b = np.sum(data[:,1])
        return a,b
    
    def plot_tot(self, n_rvs=2000, pi=0, pf=1):
        """
        plot the posterior distribution for the total result

        Args:
            n_rvs (int, optional): number of random value for the histogram. Defaults to 1000.
            pi (int, optional): lower limit for p. Defaults to 0.
            pf (int, optional): upper limit for p. Defaults to 1.
        """       

        p_pts = np.linspace(pi,pf,1000)
        data = np.array( self.data, dtype="object")
        a = np.sum(data[:,0])
        b = np.sum(data[:,1])

        post = beta(a, b)

        plot_functions.plot_tot(post,n_rvs, p_pts, xlabel="Bernoulli probabilty")

    
    def plot_exp(self, type="1D", n_pdf=1000, pi=0, pf=1):
        """
        plot "cumulative" posteriors

        Args:
            type (str, optional): one dimensional plot with x=p and y=pdf 
                                  or 2 dimensional plot with x=exp and y=p_range and z=pdf. 
                                  Defaults to "1D".
            n_pdf (int, optional): Number of points on the x axis. Defaults to 1000.
            pi (int, optional): lower limit for p. Defaults to 0.
            pf (int, optional): upper limit for p. Defaults to 1.
        """

        p_pts = np.linspace(pi,pf,n_pdf)
        data = np.array( self.data, dtype="object")
        n_exp = len(self.data)

        # cumulative hits and fails
        cumsum_alpha = np.cumsum(data[:,0])
        cumsum_beta = np.cumsum(data[:,1])
            
        exp = np.arange(0, n_exp)
        zip_post_para = zip(cumsum_alpha, cumsum_beta)

        plot_functions.plot_cum_post(beta, zip_post_para, exp, p_pts, n_pdf, post_para_label="Bernoulli probabilty", type=type)
            
    
    def plot_anim(self, pi=0, pf=1, n_pdf=1000, interval=None, list_hdi=[95,90,80,60]):
        """
        Create an animation for the evolution of the posterior

        Args:
            pi (int, optional): lower limit for the Bernoulli probability. Defaults to 0.
            pf (int, optional): upper limit for the Bernoulli probability. Defaults to 1.
            n_pdf (int, optional): number of pts for the Bernoulli probabilty. Defaults to 1000.
            interval (float, optional): time in ms between frames. Defaults to None.
            list_hdi (list, optional): list of hdi's to be displayed. Defaults to [95,90,80,60].

        Returns:
            pyplot.animate.Funcanimation
        """
        
        data = np.array(self.data, dtype="object")
        
        # cumulative hits and fails
        cumsum_alpha = np.cumsum(data[:,0])
        cumsum_beta = np.cumsum(data[:,1])

        post_para = [(ca,cb) for ca,cb in zip(cumsum_alpha, cumsum_beta)]

        return plot_functions.plot_anim(beta, post_para, pi, pf, 
                model_para_label="Bernoulli probability", list_hdi=list_hdi, n_pdf=n_pdf, interval=interval)






    