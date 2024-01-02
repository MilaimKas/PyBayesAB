
import numpy as np

from scipy.stats import gamma, poisson, nbinom

from PyBayesAB import helper
from PyBayesAB import plot_functions

N_SAMPLE=5000
N_PTS = 1000
N_BINS = 40

class BaysPoisson:
    def __init__(self, prior_a=1, prior_b=1) -> None:
        """
        Class for:
        - likelihood = Poisson
        - prior and posterior = Gamma

        Contains ploting and animation function
        """

        self.dataA = []
        self.dataB = []

        self.prior_a = prior_a
        self.prior_b = prior_b

    def add_experiment(self, n_events, group="A"):
        """
        add 1 interval with nbr of event

        Args:
            n_events (int): number of occurences or events for one interval
        """
        if group == "A":
            self.dataA.append([n_events])
        elif group == "B":
            self.dataB.append([n_events])
        else:
            raise ValueError("Group must be either 'A' or 'B'")
    
    def add_rand_experiment(self,n,mu, group="A"):
        """
        add n intervals with random nbr of events ([n_events])
        taken from a Poisson distribution with meam mu

        Args:
            n (in): number of intervals
            mu (float): Poisson mean
        """
        n_events = poisson.rvs(mu, size=n)
        if group == "A":
            self.dataA.extend(n_events)
        elif group == "B":
            self.dataB.extend(n_events)
    
    def post_pred(self, group="A"):
        """
        returns the posterior predictve distribution which gives the probabilty for the next observation
        p(x|X) where x = new observation and X = all data collected so far

        Returns:
            scipy.stats.nbinom: posterior predictive distribution
        """
        a,b = self.post_parameters(group=group)
        return nbinom(a,b/(1+b))
    
    def post_parameters(self, group="A", data=None):
        """
        return the parameters for the gamma posterior given the data
        """
        if data is None:
            if group == "A":
                data = self.dataA
            elif group == "B":
                data = self.dataB
            else:
                raise ValueError("Group must be either 'A' or 'B'")
        a = sum(data)+self.prior_a
        b = len(data)+self.prior_b

        return a,b

    def make_rvs(self, data=None, group="A", N_sample=N_SAMPLE):
        """
        Return a array of random value samples from a gamma distribution.

        Args:
            data (_type_, optional): _description_. Defaults to None.
            group (str, optional): _description_. Defaults to "A".
            N_sample (_type_, optional): _description_. Defaults to N_SAMPLE.

        Returns:
            _type_: _description_
        """
        if data is None:
            if group == "A":
                data = self.dataA
            elif group == "B":
                data = self.dataB
            else:
                raise ValueError("group must be either 'A' or 'B'")
        a,b = self.post_parameters(group=group, data=data)
        return gamma.rvs(a, scale=1/b, size=N_sample)
    
    def make_pdf(self, data=None, group="A", para_range=[0,50], N_pts=N_PTS):
        """
        Return N_pts values of the gamma posterior for the given mu range.

        Args:
            data (_type_, optional): _description_. Defaults to None.
            group (str, optional): _description_. Defaults to "A".
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.
            N_pts (_type_, optional): _description_. Defaults to N_PTS.

        Returns:
            np.array, np.array: x values, y values
        """
        if data is None:
            if group == "A":
                data = self.dataA
            elif group == "B":
                data = self.dataB
            else:
                raise ValueError("group must be either 'A' or 'B'")
        a,b = self.post_parameters(group=group, data=data)
        mu_pts = np.linspace(para_range[0], para_range[1], N_pts)
        return mu_pts, gamma.pdf(mu_pts, a, scale=1/b)

    def make_cum_post_para(self, group="A", data=None):
        """_summary_

        Args:
            group (str, optional): _description_. Defaults to "A".

        Returns:
            _type_: _description_
        """
        
        if data is None:
            if group == "A":
                data = self.dataA
            elif group == "B":
                data = self.dataB
            else:
                raise ValueError("group must be either 'A' or 'B'")
        else:
            data = np.array(data, dtype="object")

        # cumulative alpha and beta value
        a_cum = np.cumsum(data)
        b_cum = np.zeros(len(data)+1)
        b_cum[0] = self.prior_b
        b_cum[1:] = np.arange(1, len(data)+1)

        return a_cum, b_cum
    
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
                                    group, "Poisson mean")
    

    def plot_exp(self, type="1D", n_pdf=N_PTS, n_rvs=N_SAMPLE, mu_range=None, group="A"):
        """
        plot "cumulative" posteriors

        Args:
            type (str, optional): one dimensional ("1D") plot with x=p and y=pdf 
                                    or 2 dimensional ("2D) plot with x=exp and y=p_range and z=pdf. 
                                    Defaults to "1D".
            n_pdf (int, optional): Number of points on the x axis. Defaults to 1000.
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.

            """
            
        gamma_post = helper.GammaCustom

        N_exp = len(self.dataA)

        return plot_functions.plot_helper(self.make_rvs, self.make_cum_post_para, gamma_post, 
                group, type, N_exp, 
                n_pdf, n_rvs,
                "Poisson mean", "\mu",
                xrange=mu_range)
        
    def plot_anim(self, mu_range=None, n_pdf=N_PTS, n_rvs=N_SAMPLE, interval=None, 
                  list_hdi=[95,90,80,60], group="A"):
        """
        Create an animation for the evolution of the posterior

        Args:
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.
            n_pdf (int, optional): number of pts for the Poisson means. Defaults to 1000.
            interval (float, optional): time in ms between frames. Defaults to None.
            list_hdi (list, optional): list of hdi's to be displayed. Defaults to [95,90,80,60].

        Returns:
            pyplot.animate.Funcanimation
        """

        gamma_post = helper.GammaCustom
    
        if (group == "A") or (group == "B"):
            post_para = [(a,b) for a,b in zip(*self.make_cum_post_para(group=group))]
            if mu_range is None:
                rvs = self.make_rvs(group=group, N_sample=n_rvs)
                mu_range = helper.make_range(rvs)
            return plot_functions.plot_anim_pdf(gamma_post, post_para, mu_range, 
                                                model_para_label="Poisson mean", 
                                                list_hdi=list_hdi, n_pdf=n_pdf, interval=interval)

        elif group == "diff":
            A_a, A_b = self.make_cum_post_para(group="A")
            B_a, B_b = self.make_cum_post_para(group="B")
            rvs_list = []
            rvs_diff = self.make_rvs()-self.make_rvs(group="B", N_sample=n_rvs)
            if mu_range is None:
                mu_range = helper.make_range(rvs_diff)
            for aa,ab,ba,bb in zip(A_a, A_b, B_a, B_b):
                rvs = gamma(aa, scale=1/ab).rvs(size=N_SAMPLE)-gamma(ba, scale=1/bb).rvs(size=N_SAMPLE)
                rvs_list.append(rvs)
            
            return plot_functions.plot_anim_rvs(rvs_list, mu_range=mu_range)