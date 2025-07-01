
import numpy as np

from scipy.stats import dirichlet
from scipy.stats import multinomial

from PyBayesAB import N_SAMPLE, N_PTS

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  


class MultinomMixin:
    """    Class for:
    - Likelihood  = Multinomial
    - prior and posterior = Dirichlet
    - model parameter = mu (probabilities of each category)
    """

    def __init__(self, prior=None):

        if prior is None:
            prior = np.ones(2)  # Default prior for two categories

    def make_default_range(self, alpha=None):
        """
        Make a default range for the Dirichlet posterior given alpha.

        Args:
            alpha (array, optional): alpha parameter for the Dirichlet distribution. Defaults to None.
        """
        if alpha is None:
            alpha = self.prior
        if isinstance(alpha, list):
            alpha = np.array(alpha)
        if len(alpha) == 0:
            raise ValueError("Alpha must have at least one category")
        
        # Calculate the mean and standard deviation of the Dirichlet distribution
        mu_mean = alpha / np.sum(alpha)
        mu_std = np.sqrt(mu_mean * (1 - mu_mean) / (np.sum(alpha) + 1))
        
        # Create a range around the mean with 3 standard deviations
        lower = max(0, mu_mean - 3 * mu_std)
        upper = min(1, mu_mean + 3 * mu_std)
        
        return [lower, upper]

    def add_rand_experiment(self, n, mu,, group="A"):
        """
        Add a random experiment to the data for group A or B.
        n: number of trials
        mu: probabilities of each category
        group: "A" or "B"
        """
        if group not in ["A", "B"]:
            raise ValueError("Group must be 'A' or 'B'")
        if not isinstance(mu, np.ndarray):
            mu = np.array(mu)
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        if len(mu) == 0:
            raise ValueError("mu must have at least one category")
        if n <= 0:
            raise ValueError("n must be greater than 0")
        
        data = multinomial.rvs(n=n, p=mu)
        if group == "A":
            self.dataA.append(data)
        else:
            self.dataB.append(data)
    
    def post_pred(self, group="A"):
        """
        returns the posterior predictve distribution which gives the probabilty for the next observation
        p(x|X) where x = new observation and X = all data collected so far
        """
        return NotImplementedError

    def post_parameters(self, group="A", data=None):
        """
        Returns the posterior parameters for the Dirichlet distribution, given group.
        """
        if group not in ["A", "B"]:
            raise ValueError("Group must be 'A' or 'B'")
        
        if data is None:
            data = self.dataA if group == "A" else self.dataB
        
        if len(data) == 0:
            raise ValueError(f"No data available for group {group}")
        
        # Sum the counts for each category
        counts = np.sum(data, axis=0)
                
        return counts + self.prior
    
    def get_parameters(self, group,  parameters=None, data=None):
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Gamma posterior needs 2 parameters: alpha and beta")
            else:
                alpha = parameters
        else:
            alpha = self.post_parameters(group=group, data=data)
        return alpha
    
    def make_rvs(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        """
        Generate random variates from the Dirichlet distribution.

        Args:
            parameters (array, optional): alpha array parameter. Defaults to None.
        """
        alpha = self.get_parameters(group, parameters, data)
        return dirichlet.rvs(alpha, size=N_sample)
    
    def make_pdf(self, parameters=None, data=None, group="A", p_pts=None, para_range=None, var="mu"):
        """
        Return N_pts values of the Dirichlet posterior for the given mu range.

        Args:
            data (array, optional): Data to use for the calculation. Defaults to None.
            group (str, optional): "A" or "B". Defaults to "A".
            p_pts (array, optional): Points at which to evaluate the PDF. Defaults to None.
            para_range (list, optional): Range for mu. Defaults to None.
            var (str, optional): Variable to use. Defaults to "mu".

        Returns:
            np.array: PDF values at the specified points.
        """
        if group not in ["A", "B"]:
            raise ValueError("Group must be either 'A' or 'B' for pdf calculation.")
        
        alpha = self.get_parameters(group, parameters, data)
        
        if p_pts is None:
            if para_range is None:
                para_range = self.make_default_range(alpha=alpha)
            p_pts = np.linspace(para_range[0], para_range[1], N_PTS)
        
        return dirichlet.pdf(p_pts, alpha)
        
    def make_cum_post_para(self, group="A"):
        data = self.return_data(group)
        alpha_cum = self.prior
        alphas = [alpha_cum]    
        for d in data:
            alpha_cum += d
            alphas.append(alpha_cum)
        return np.array(alphas)

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        """
        Generate cumulative posterior distribution for the specified group.
        """
        # create list of rvs and pdf
        alpha_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        if para_range is None:
            para_range = self.make_default_mu_range(alpha_cum)
        p_pts = np.linspace(para_range[0], para_range[1], N_pts)
        for a in alpha_cum:
            rvs = dirichlet.rvs(a, size=N_sample)
            pdf = dirichlet.pdf(p_pts, a)
            rvs_data.append(rvs)
            pdf_data.append(pdf)
        return p_pts, rvs_data, pdf_data
    
    
class BaysMultinomial(MultinomMixin, BayesianModel, PlotManager):
    def __init__(self, prior=None):
        BayesianModel.__init__(self)
        MultinomMixin.__init__(self, prior=prior)