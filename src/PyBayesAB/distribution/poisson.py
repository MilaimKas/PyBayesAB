import numpy as np

from scipy.stats import gamma, poisson, nbinom

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  

from PyBayesAB.config import N_SAMPLE, N_PTS

PARA_RANGE=[0, np.inf]

class PoissonMixin:
    """
    Class for:
    - Likelihood  = Poisson
    - prior and posterior = Gamma
    - model parameter = lambda (rate)
    """
            
    def __init__(self, prior=[1,1]) -> None:

        if len(prior) != 2:
            raise ValueError(" Number of parameters for gamma prior is 2 (alpha and beta)")
        
        self.dataA = []
        self.dataB = []

        self.prior = prior
        self.parameter_name = "Poisson mean"

    def _get_parameters(self, parameters, group, data):
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Gamma posterior needs 2 parameters: alpha and beta")
            else:
                a,b = parameters
        else:
            a,b = self.post_parameters(group=group, data=data)
        return a, b
    
    def make_default_mu_range(self, a, b):
        """
        make a default mu range for the gamma posterior given the parameters a and b. If list or array is given, use the last element.
        """
        # calculate the mean and standard deviation of all poserior distributions
        if isinstance(a, (list, np.ndarray)):
            a = a[-1]
        if isinstance(b, (list, np.ndarray)):
            b = b[-1]
        mu_mean = a / b
        mu_std = np.sqrt(a) / b
        # create a range around the mean with 3 standard deviations
        lower =  mu_mean - 3*mu_std
        upper = mu_mean + 3*mu_std
        # ensure the range is not negative  
        if lower < 0:
            lower = 0
        if upper < lower:
            upper = lower + 1
        return [lower, upper]
    
    def add_rand_experiment(self, n, mu, group="A"):
        """
        add n intervals with random nbr of events ([n_events])
        taken from a Poisson distribution with meam mu

        Args:
            n (in): number of intervals
            mu (float): Poisson mean
        """
        self.add_experiment(poisson.rvs(mu, size=n), group=group)
    
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
            data = np.concatenate(self.return_data(group)).ravel()

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
        a,b = self._get_parameters(parameters, group, data)
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
        if group not in ["A", "B"]:
            raise ValueError("Group must be either 'A' or 'B' for pdf calculation.")
        a,b = self._get_parameters(parameters, group, data)
        if p_pts is None:
            if para_range is None:
                para_range = self.make_default_mu_range(a, b)
            p_pts = np.linspace(para_range[0], para_range[1], N_PTS)
        return gamma.pdf(p_pts, a, scale=1/b)

    def make_cum_post_para(self, group="A"):
        data = self.return_data(group)

        a_cum = self.prior[0]  # initial alpha from prior
        b_cum = self.prior[1]  # initial beta from prior
        alphas = [a_cum]
        betas = [b_cum]
        # cumulative alpha and beta value
        for i in range(len(data)):
            a_cum += sum(data[i])  # sum of events in each group
            b_cum += len(data[i])  # number of intervals in each group  
            alphas.append(a_cum)
            betas.append(b_cum)
        # convert to numpy arrays for consistency
        alphas = np.array(alphas)
        betas = np.array(betas)
        return alphas, betas

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        # create list of rvs and pdf
        a_cum, b_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        if para_range is None:
            para_range = self.make_default_mu_range(a_cum, b_cum)
        p_pts = np.linspace(para_range[0], para_range[1], N_pts)
        for a,b in zip(a_cum, b_cum):
            rvs_data.append(self.make_rvs(parameters=[a,b], N_sample=N_sample))
            pdf_data.append(self.make_pdf(parameters=[a,b], p_pts=p_pts))
        return p_pts, rvs_data, pdf_data

class BaysPoisson(PoissonMixin, BayesianModel, PlotManager):
    def __init__(self, prior=[1,1]):
        BayesianModel.__init__(self)
        PoissonMixin.__init__(self,  prior=prior)


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt

    # Example usage
    # Create Likelihhod = Poisson object
    Pois_test = BaysPoisson()

    # create data
    mu_A = 20
    mu_B = 21
    n_exp = 20
    for next in range(n_exp):
        n_events = np.random.randint(5, 10)
        # add random experiment for group A and B
        Pois_test.add_rand_experiment(n_events, mu_A)
        Pois_test.add_rand_experiment(n_events, mu_B, group="B")
    
    Pois_test.make_rvs()
    Pois_test.make_pdf()
    Pois_test.make_cum_posterior()

    # Generate some Poisson data as pandas dataframe
    data = pd.DataFrame({
        'group': ['A', 'B', 'A', 'B', 'B'],
        'values': [5, 7, 6, 8, 14],
        'experiment': [1, 1, 2, 2, 1]
    })  

    # create a model for Poisson data   
    Pois_test = BaysPoisson()
    # add the data
    Pois_test.add_test_result(data)
    # plot
    fig = Pois_test.plot_cum_posterior()
    plt.show()