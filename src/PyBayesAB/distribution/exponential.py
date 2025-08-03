import numpy as np

from scipy.stats import gamma, expon, lomax

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  
from PyBayesAB import helper

from PyBayesAB.config import N_SAMPLE, N_PTS

class ExponentialMixin:
    """
    Class for:
    - Likelihood  = exponential
    - prior and posterior = Gamma
    - model parameter = lambda (rate)
    """

    def __init__(self, prior=[1, 1]) -> None:
        
        if len(prior) != 2:
            raise ValueError(" Number of parameters for gamma prior is 2 (alpha and beta)")
        
        self.dataA = []
        self.dataB = []

        self.prior = prior
        self.parameter_name = "Exponential rate"

    def get_parameters(self, parameters, group, data):
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Gamma posterior needs 2 parameters: alpha and beta")
            else:
                a,b = parameters
        else:
            a,b = self.post_parameters(group=group, data=data)
        return a, b
    
    def make_default_lambd_range(self, a, b, percentile=0.9999):
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

    def add_rand_experiment(self, n, lambd, group="A"):
        """_summary_

        Args:
            n (_type_): _description_
            lambda (_type_): _description_
            group (str, optional): _description_. Defaults to "A".
        """
        for exp in expon.rvs(scale=1/lambd, size=n):
            self.add_experiment(exp, group=group)

    def post_pred(self, size=1, group="A"):
        """
        returns the posterior predictve distribution which gives the probabilty for the next observation
        p(x|X) where x = new observation and X = all data collected so far

        Returns:
            scipy.stats.lomax: posterior predictive distribution
        """
        a,b = self.post_parameters(group=group)
        return lomax.rvs(c=a, scale=1/b, size=size)
    
    def post_parameters(self, group="A", data=None):
        """
        return the parameters for the gamma posterior given the data
        """
        if data is None:
            data = self.return_data(group)
        a = len(data)+self.prior[0]
        b = sum(data)+self.prior[1]

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
    
    def make_pdf(self, parameters=None, data=None, group="A", lambd_pts=None, para_range=None):
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
        if lambd_pts is None:
            if lambd_pts is None:
                para_range = self.make_default_lambd_range(a, b)
            lambd_pts = np.linspace(para_range[0], para_range[1], N_PTS)
        return gamma.pdf(lambd_pts, a, scale=1/b)

    def make_cum_post_para(self, group="A"):
        data = self.return_data(group)
        
        # cumulative alpha and beta value
        b_cum = np.zeros(len(data)+1)
        b_cum[0] = self.prior[1]
        b_cum[1:] = data
        b_cum = np.cumsum(b_cum)

        a_cum = np.full(len(data)+1, self.prior[0])
        a_cum[1:] += np.arange(1, len(data)+1)
        return a_cum, b_cum

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        # create list of rvs and pdf
        a_cum, b_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        if para_range is None:
            para_range = self.make_default_lambd_range(a_cum[-1], b_cum[-1])
        lambd_pts = np.linspace(para_range[0], para_range[1], N_pts)
        for a,b in zip(a_cum, b_cum):
            rvs_data.append(self.make_rvs(parameters=[a,b], N_sample=N_sample))
            pdf_data.append(self.make_pdf(parameters=[a,b], lambd_pts=lambd_pts))
        return lambd_pts, rvs_data, pdf_data

class BaysExponential(ExponentialMixin, BayesianModel, PlotManager):
    def __init__(self, prior=[1, 1]):
        BayesianModel.__init__(self)
        ExponentialMixin.__init__(self,  prior=prior)