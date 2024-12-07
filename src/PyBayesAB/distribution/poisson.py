
import numpy as np

from scipy.stats import gamma, poisson, nbinom

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  

from PyBayesAB import N_SAMPLE, N_PTS

PARA_RANGE=[0, np.inf]

class PoissonMixin:
    def __init__(self, prior=[1,1]) -> None:
        """
        Class for:
        - likelihood = Poisson
        - prior and posterior = Gamma
        """

        if len(prior) != 2:
            raise ValueError(" Number of parameters for gamma prior is 2 (alpha and beta)")
        
        self.dataA = []
        self.dataB = []

        self.prior = prior
        self.parameter_name = "Poisson mean"

    def get_parameters(self, parameters, group, data):
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Gamma posterior needs 2 parameters: alpha and beta")
            else:
                a,b = parameters
        else:
            a,b = self.post_parameters(group=group, data=data)
        return a, b
    
    def make_default_mu_range(self, a, b, percentile=0.9999):
        """
        mean + variance as max
        """
        # Define the percentile bounds
        lower_percentile = 1-percentile
        upper_percentile = percentile

        # Calculate the meaningful range
        xmin = gamma.ppf(lower_percentile, a=a, scale=1/b)
        xmax = gamma.ppf(upper_percentile, a=a, scale=1/b)
        return [xmin, xmax]
    
    def add_rand_experiment(self, n, mu, group="A"):
        """
        add n intervals with random nbr of events ([n_events])
        taken from a Poisson distribution with meam mu

        Args:
            n (in): number of intervals
            mu (float): Poisson mean
        """
        n_events = poisson.rvs(mu, size=n)
        for ev in n_events:
            self.add_experiment(ev, group=group)
    
    def post_pred(self, size=1, group="A"):
        """
        returns the posterior predictve distribution which gives the probabilty for the next observation
        p(x|X) where x = new observation and X = all data collected so far

        Returns:
            scipy.stats.nbinom: posterior predictive distribution
        """
        a,b = self.post_parameters(group=group)
        return nbinom.rvs(a,b/(1+b), size=size)
    
    def post_parameters(self, group="A", data=None):
        """
        return the parameters for the gamma posterior given the data
        """
        if data is None:
            data = self.return_data(group)
        a = sum(data)+self.prior[0]
        b = len(data)+self.prior[1]

        return a, b

    def make_rvs(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        """
        Return a array of random value samples from a gamma distribution.

        Args:
            data (_type_, optional): _description_. Defaults to None.
            group (str, optional): _description_. Defaults to "A".
            N_sample (_type_, optional): _description_. Defaults to N_SAMPLE.

        Returns:
            _type_: _description_
        """
        a,b = self.get_parameters(parameters, group, data)
        return gamma.rvs(a, scale=1/b, size=N_sample)
    
    def make_pdf(self, parameters=None, data=None, group="A", p_pts=None, para_range=None):
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
        a,b = self.get_parameters(parameters, group, data)
        if p_pts is None:
            if para_range is None:
                para_range = self.make_default_mu_range(a, b)
            p_pts = np.linspace(para_range[0], para_range[1], N_PTS)
        return gamma.pdf(p_pts, a, scale=1/b)

    def make_cum_post_para(self, group="A"):
        data = self.return_data(group)
        # cumulative alpha and beta value
        a_cum = np.cumsum(data)
        b_cum = np.zeros(len(data)+1)
        b_cum[0] = self.prior[1]
        b_cum[1:] = np.arange(1, len(data)+1)
        return a_cum, b_cum

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        # create list of rvs and pdf
        a_cum, b_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        if para_range is None:
            para_range = self.make_default_mu_range(a_cum[-1], b_cum[-1])
        p_pts = np.linspace(para_range[0], para_range[1], N_pts)
        for a,b in zip(a_cum, b_cum):
            rvs_data.append(self.make_rvs(parameters=[a,b], N_sample=N_sample))
            pdf_data.append(self.make_pdf(parameters=[a,b], p_pts=p_pts))
        return p_pts, rvs_data, pdf_data


class BaysPoisson(PoissonMixin, BayesianModel, PlotManager):
    def __init__(self, prior=[1,1]):
        BayesianModel.__init__(self)
        PoissonMixin.__init__(self,  prior=prior)