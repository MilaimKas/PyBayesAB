

import numpy as np

from scipy.stats import beta
from scipy.stats import bernoulli

import matplotlib.pyplot as plt

import src.helper as helper
import src.plot_functions as plot_functions

N_SAMPLE = 5000
N_PTS = 1000
N_BINS = 20


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
        self.dataB = []
        self.dataB.append(self.prior)
    
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
            self.data.append([hits,n-hits])
        elif group == "B":
            self.dataB.append([hits,n-hits])

    def post_pred(self, data=None, group="A"):
        """
        returns the probabilty that the next data points will be a hit

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
                data = self.data
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
    
    def make_pdf(self, data=None, group="A", p_range=[0,1], N_pts=N_PTS):
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
        p_pts = np.linspace(p_range[0], p_range[1], N_pts)
        return p_pts, beta.pdf(p_pts, a, b)
    
    def make_cum_post_para(self, group="A"):
        """_summary_

        Args:
            group (str, optional): _description_. Defaults to "A".

        Returns:
            _type_: _description_
        """
        
        if group == "A":
            data = self.data
        elif group == "B":
            data = self.dataB

        data = np.array(data, dtype="object")

        # cumulative hits and fails
        cumsum_alpha = np.cumsum(data[:,0])
        cumsum_beta = np.cumsum(data[:,1])
            
        return cumsum_alpha, cumsum_beta

    def plot_tot(self, data=None, group="A", n_rvs=N_SAMPLE, p_range=None, n_pts=N_PTS):
        """
        plot the posterior distribution for the total result

        Args:
            n_rvs (int, optional): number of random value for the histogram. Defaults to 1000.
            prange (list, optional): [lower, upper] limit for p. Defaults to None.
        """       
        
        if data is None:
            
            if (group == "A") or (group == "B"):
                rvs = self.make_rvs(group=group, N_sample=n_rvs)
                if p_range is None:
                    p_range = [np.min(rvs), np.max(rvs)]
                model_para_pts, post = self.make_pdf(group=group, p_range=p_range)
                fig = plot_functions.plot_tot([rvs], model_para_pts, [post], labels=[group], xlabel="Bernoulli probabilty")
            
            elif group == "diff":
                rvs_A = self.make_rvs(group="A", N_sample=n_rvs) 
                rvs_B = self.make_rvs(group="B", N_sample=n_rvs)
                rvs_diff = rvs_A-rvs_B
                if p_range is None:
                    p_range = [np.min(rvs_diff), np.max(rvs_diff)]
                model_para_pts = np.linspace(p_range[0],p_range[1],n_pts)
                fig = plot_functions.plot_tot([rvs_diff],model_para_pts, labels=["A-B"], xlabel="Bernoulli probabilty")
            
            elif group == "AB":
                rvs_A = self.make_rvs(group="A", N_sample=n_rvs)
                rvs_B = self.make_rvs(group="B", N_sample=n_rvs)
                rvs_tmp = np.concatenate((rvs_A, rvs_B))
                if p_range is None:
                    p_range = [np.min(rvs_tmp), max(rvs_tmp)]
                model_para_pts, post_A = self.make_pdf(group="A", p_range=p_range)
                _, post_B = self.make_pdf(group="B", p_range=p_range)
                fig = plot_functions.plot_tot([rvs_A, rvs_B], model_para_pts, [post_A, post_B],
                                              labels=["A", "B"], 
                                              xlabel="Bernoulli probabilty")    
            else:
                raise SyntaxError("group can only be A,B,diff or AB")
        
        else:
            raise NotImplementedError

        return fig
    
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

        exp = np.arange(1, len(self.data)+1)

        if (group == "A") or (group == "B"):
            if p_range is None:
                rvs = self.make_rvs(group=group)
                p_range = [np.min(rvs), np.max(rvs)]
            p_pts = np.linspace(p_range[0],p_range[1],n_pdf)
            zip_post_para = [zip(*self.make_cum_post_para(group=group))]
            labels = [group]
            if type == "2D":
                fig = plot_functions.plot_cum_post_2D_pdf(beta, zip_post_para, labels, 
                                        exp, p_pts, 
                                        post_para_label="Bernoulli probabilty")
            elif type == "1D":
                fig = plot_functions.plot_cum_post_1D_pdf(beta, zip_post_para, labels, 
                                                          exp, p_pts, 
                                                          post_para_label="Bernoulli probabilty")
                
        elif group == "AB":
            zip_post_para = [zip(*self.make_cum_post_para(group="A")), 
                                zip(*self.make_cum_post_para(group="B"))]
            labels = ["A", "B"]
            if p_range is None:
                rvs = np.concatenate((self.make_rvs(),self.make_rvs(group="B")))
                p_range = [np.min(rvs), np.max(rvs)]
            p_pts = np.linspace(p_range[0],p_range[1],n_pdf)
            if type == "2D":
                fig = plot_functions.plot_cum_post_2D_pdf(beta, zip_post_para, labels, 
                                        exp, p_pts, 
                                        post_para_label="Bernoulli probabilty")
            elif type == "1D":
                fig = plot_functions.plot_cum_post_1D_pdf(beta, zip_post_para, labels, 
                                        exp, p_pts, 
                                        post_para_label="Bernoulli probabilty")
                
        elif group == "diff":
            A_a, A_b = self.make_cum_post_para(group="A")
            B_a, B_b = self.make_cum_post_para(group="B")
            rvs_diff = beta(a=A_a[0], b=A_b[0]).rvs(size=n_rvs)-beta(a=B_a[0], b=B_b[0]).rvs(size=n_rvs)
            if p_range is None:
                p_range = [np.min(rvs_diff), max(rvs_diff)]
            p_pts = np.linspace(p_range[0],p_range[1],n_pdf)
            hist_list = []
            rvs_list = []
            for aa,ab,ba,bb in zip(A_a, A_b, B_a, B_b):
                rvs = beta(a=aa, b=ab).rvs(size=n_rvs)-beta(a=ba, b=bb).rvs(size=n_rvs)
                hist = np.histogram(rvs, bins=N_BINS, range=p_range, density=True)[0]
                hist_list.append(hist)
                rvs_list.append(rvs)

            if type == "2D":
                hist_arr = np.array(hist_list)
                fig = plot_functions.plot_cum_post_2D_rvs(hist_arr, p_range)
            elif type == "1D":
                fig = plot_functions.plot_cum_post_1D_rvs([rvs_list], exp, 
                                                          model_para_pts=p_pts, labels=["A-B"],
                                                          post_para_label="Difference of Bernouli probability")

        return fig
            
    
    def plot_anim(self, p_range=None, n_pdf=1000, interval=None, list_hdi=[95,90,80,60], group="A"):
        """
        Create an animation for the evolution of the posterior

        Args:
            n_pdf (int, optional): number of pts for the Bernoulli probabilty. Defaults to 1000.
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





    