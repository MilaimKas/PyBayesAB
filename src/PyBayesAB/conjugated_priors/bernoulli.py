

import numpy as np

from scipy.stats import beta
from scipy.stats import bernoulli

import matplotlib.pyplot as plt

from PyBayesAB import helper
from PyBayesAB import plot_functions
from PyBayesAB import bayesian_functions as bf

from PyBayesAB import N_BINS, N_SAMPLE, COLORS, N_PTS, FIGSIZE


class BaysBernoulli:
    def __init__(self, prior_type='Bayes-Laplace'):
        '''
        Class for:
        - likelihood = Bernoulli
        - prior and posterior = Beta

        Contains plotting and animation function
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
        self.dataA = []
        self.dataA.append(self.prior)

        # group B (for AB test)
        self.dataB = []
        self.dataB.append(self.prior)

        # dataframe
        self.data = None
    
    def add_data(self, df, group_col_name="group"):
        """
        Add data as pandas dataframe with time sequence as index

        Args:
            df (pandas.DataFrame): must contain a column to distinguishes both A and B group
        """

        if not all(x in df[group_col_name].unique() for x in ['A', 'B']): 
            raise ValueError("the {} column must contains at least one value for each group A and B".format(group_col_name))

        self.data = df
        self.dataA.append(df[df[group_col_name]=="A"].drop(columns=group_col_name).values.tolist())
        self.dataB.append(df[df[group_col_name]=="B"].drop(columns=group_col_name).values.tolist())

    
    def add_experiment(self, hits, fails, group="A"):
        """
        add an experiment ([N_hits, N_fails]) to the data

        Args:
            hits (int): number of success
            fails (int): number of failures
        """
        if group == "A":
            self.data.append([hits,fails])
        elif group == "B":
            self.dataB.append([hits,fails])

    def add_rand_experiment(self,n,p, group="A"):
        """
        add an experiment with random trials ([N_hits, N_fails])
        taken from a Bernoulli distribution with probability p

        Args:
            n (int): number of trilas
            p (float): parameter for the Bernoulli distribution
        """
        hits = np.count_nonzero(bernoulli.rvs(p, size=n))
        if group == "A":
            self.dataA.append([hits,n-hits])
        elif group == "B":
            self.dataB.append([hits,n-hits])

    def post_pred(self, data=None, group="A"):
        """
        returns the probability that the next data points will be a hit

        Returns:
            float: posterior predictive for a hits
        """
        a,b = self.post_parameters(data=data, group=group)
        return a/(a+b)
    
    def post_parameters(self, data=None, group="A"):
        """
        return the parameter of the Beta posterior alpha and beta given the data

        Returns:
            tuple: alpha and beta value for the Beta posterior
        """

        if data is None:
            if group == "A":
                data = self.dataA
            elif group == "B":
                data = self.dataB

        data = np.array(data, dtype="object")
        a = np.sum(data[:,0])
        b = np.sum(data[:,1])

        return a,b

    def make_rvs(self,data=None, group="A", N_sample=N_SAMPLE):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            group (str, optional): _description_. Defaults to "A".
            N_sample (_type_, optional): _description_. Defaults to N_SAMPLE.

        Returns:
            _type_: _description_
        """
        a,b = self.post_parameters(data=data, group=group)
        return beta.rvs(a,b,size=N_sample)
    
    def make_pdf(self, data=None, group="A", para_range=[0,1], N_pts=N_PTS):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            group (str, optional): _description_. Defaults to "A".
            pi (int, optional): _description_. Defaults to 0.
            pf (int, optional): _description_. Defaults to 1.
            N_pts (_type_, optional): _description_. Defaults to N_PTS.

        Returns:
            _type_: _description_
        """
        a,b = self.post_parameters(data=data, group=group)
        p_pts = np.linspace(para_range[0], para_range[1], N_pts)
        return p_pts, beta.pdf(p_pts, a, b)
    
    def make_cum_post_para(self, group="A"):
        """_summary_

        Args:
            group (str, optional): _description_. Defaults to "A".

        Returns:
            _type_: _description_
        """
        
        if group == "A":
            data = self.dataA
        elif group == "B":
            data = self.dataB

        data = np.array(data, dtype="object")

        # cumulative hits and fails
        cumsum_alpha = np.cumsum(data[:,0])
        cumsum_beta = np.cumsum(data[:,1])
            
        return cumsum_alpha, cumsum_beta
    
    def prob_best(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        rvs_A = self.make_rvs(group="A")
        rvs_B = self.make_rvs(group="B")
        
        return bf.prob_best(rvs_A-rvs_B)

    def plot_tot(self, group="A"):
        """
        plot the posterior distribution for the total result
        Either A or B, both A and B, or their difference

        Args:
            n_rvs (int, optional): number of random values for the histogram. Defaults to 1000.
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.
            group (str, optional): can be 'A', 'B', 'diff' or 'AB'
        """
        return plot_functions.make_plot_tot(self.make_rvs, self.make_pdf, 
                                    group, "Bernoulli probability")

    
    def plot_exp(self, group="A", type="1D", n_pdf=N_PTS, p_range=None, n_rvs=N_SAMPLE):
        """
        plot "cumulative" posteriors

        Args:
            type (str, optional): one dimensional plot with x=p and y=pdf 
                                  or 2 dimensional plot with x=exp and y=p_range and z=pdf. 
                                  Defaults to "1D".
            n_pdf (int, optional): Number of points on the x axis. Defaults to 1000.
            p_range (list, optional): lower, upper] limit for p. Defaults to None.
        """

        N_exp = len(self.dataA)

        return plot_functions.plot_helper(self.make_rvs, self.make_cum_post_para, beta, 
                group, type, N_exp, 
                n_pdf, n_rvs,
                "Bernoulli probability", "p",
                xrange=p_range)
            
    
    def plot_anim(self, p_range=None, n_pdf=1000, interval=None, list_hdi=[95,90,80,60], group="A"):
        """
        Create an animation for the evolution of the posterior

        Args:
            n_pdf (int, optional): number of pts for the Bernoulli probability. Defaults to 1000.
            interval (float, optional): time in ms between frames. Defaults to None.
            list_hdi (list, optional): list of hdi's to be displayed. Defaults to [95,90,80,60].
            p_range (list, optional): lower, upper] limit for p. Defaults to None.

        Returns:
            pyplot.animate.Funcanimation
        """
        
        if (group == "A") or (group == "B"):
            post_para = [(a,b) for a,b in zip(*self.make_cum_post_para(group=group))]
            if p_range is None:
                rvs = self.make_rvs(group=group)
                p_range = [np.min(rvs),np.max(rvs)]
            return plot_functions.plot_anim_pdf(beta, post_para, p_range, 
                model_para_label="Bernoulli probability", list_hdi=list_hdi, n_pdf=n_pdf, interval=interval)

        elif group == "diff":
            A_a, A_b = self.make_cum_post_para(group="A")
            B_a, B_b = self.make_cum_post_para(group="B")
            rvs_list = []
            rvs = self.make_rvs()-self.make_rvs(group="B")
            p_range = [np.min(rvs), np.max(rvs)]
            for aa,ab,ba,bb in zip(A_a, A_b, B_a, B_b):
                rvs = beta(a=aa, b=ab).rvs(size=N_SAMPLE)-beta(a=ba, b=bb).rvs(size=N_SAMPLE)
                rvs_list.append(rvs)
            
            return plot_functions.plot_anim_rvs(rvs_list, p_range)





    