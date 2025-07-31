
import numpy as np

from scipy.stats import dirichlet, beta
from scipy.stats import multinomial

from PyBayesAB import N_SAMPLE, N_PTS

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  

import matplotlib.pyplot as plt


class MultinomMixin:
    """    Class for:
    - Likelihood  = Multinomial
    - prior and posterior = Dirichlet
    - model parameter = mu (probabilities of each category)
    """

    def __init__(self, prior):

        self.prior = prior

    def make_default_p_range(self, alphas=None):
        """
        Make a default range for each categories of the Dirichlet posterior given alpha.

        Args:
            alpha (array, optional): list of cumulative alpha parameter for the Dirichlet distribution. Defaults to None.
        
            Returns:
                list: [p_min, p_max] range for the Dirichlet posterior, where p_min and p_max are arrays of the minimum and maximum probabilities for each category.
        """

        if alphas is None:
            alphas = self.prior
        if len(alphas) == 0:
            raise ValueError("Alpha must have at least one category")
        
        # Calculate the mean and standard deviation of the Dirichlet distribution
        alpha = alphas[-1]  # Use the last alpha for the range
        p_mean = alpha / np.sum(alpha)
        p_std = np.sqrt((p_mean * (1 - p_mean)) / (np.sum(alpha) + 1))
        p_max = p_mean + 3 * p_std
        p_min = p_mean - 3 * p_std

        return [p_min, p_max]

    def add_rand_experiment(self, n, p, group="A"):
        """
        Add a random experiment to the data for group A or B.
        n: number of trials
        mup: probabilities of each category
        group: "A" or "B"
        """
        if group not in ["A", "B"]:
            raise ValueError("Group must be 'A' or 'B'")
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        if len(p) == 0:
            raise ValueError("mu must have at least one category")
        if n <= 0:
            raise ValueError("n must be greater than 0")
        
        data = multinomial.rvs(n=n, p=p)
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
            if len(parameters) != len(self.prior):
                raise ValueError("Dirichlt parameter length  must be equal to prior length")
            else:
                alphas = parameters
        else:
            alphas = self.post_parameters(group=group, data=data)
        return alphas
    
    def make_rvs(self, category_idx=0, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        """
        Generate random variates from the marginal Beta distribution: category_idx against all other categories.

        Args:
            parameters (array, optional): alpha array parameter. Defaults to None.
        """
        alpha = self.get_parameters(group, parameters, data)
        a_i = alpha[category_idx]
        a_rest = alpha.sum() - a_i
        return beta.rvs(a_i, a_rest, size=N_sample)
    
    def make_rvs_dirichlet(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        """
        Generate random variates from the Dirichlet distribution.

        Args:
            parameters (array, optional): alpha array parameter. Defaults to None.
        """
        alpha = self.get_parameters(group, parameters, data)
        return dirichlet.rvs(alpha, size=N_sample)
    
    def make_pdf(self, category_idx=0, parameters=None, data=None, group="A", p_pts=None, para_range=None, var="mu"):
        """
        Return N_pts values of the marginal Beta posterior for the given mu range. Category_idx against all other categories.

        Args:
            category_idx (int): Index of the category for which to calculate the marginal Beta PDF. Defaults to 0 (first category).
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
                para_range = self.make_default_p_range(alpha=alpha)
                para_min = para_range[0][category_idx]
                para_max = para_range[1][category_idx]
            
        # take the range for the category
        p_pts = np.linspace(para_min, para_max, N_PTS)
        a_i = alpha[category_idx]
        a_rest = alpha.sum() - a_i
        return beta.pdf(p_pts, a_i, a_rest)
        
    def make_cum_post_para(self, group="A"):
        data = self.return_data(group)
        alpha_cum = self.prior
        alphas = [alpha_cum]    
        for d in data:
            alpha_cum += d
            alphas.append(alpha_cum)
        return np.array(alphas)

    def make_cum_posterior(self, category_idx=0, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        """
        Generate cumulative posterior distribution of the marginal Beta distribution for the specified group and category.
        """
        # create list of rvs and pdf
        alpha_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        if para_range is None:
            para_range = self.make_default_p_range(alpha_cum)
        p_pts = np.linspace(para_range[0][category_idx], para_range[1][category_idx], N_pts)
        # generate rvs and pdf for each alpha_cum
        for a in alpha_cum:
            a_i = a[category_idx]
            a_rest = a.sum() - a_i
            rvs = beta.rvs(a_i, a_rest, size=N_sample)
            pdf = beta.pdf(p_pts, a_i, a_rest)
            rvs_data.append(rvs)
            pdf_data.append(pdf)
        return p_pts, rvs_data, pdf_data
    
    def make_cum_posterior_dirichlet(self, group="A", N_sample=N_SAMPLE):
        """
        Generate cumulative posterior distribution of the Dirichlet distribution for the specified group.
        """
        # create list of rvs and pdf
        alpha_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        # generate rvs for each alpha_cum
        for a in alpha_cum:
            rvs = dirichlet.rvs(a, size=N_sample)
            rvs_data.append(rvs)
        return rvs_data

    def plot_dirichlet_rvs(self, group="A", N_sample=N_SAMPLE):
        """
        Plot random variates from the Dirichlet distribution for the specified group.
        """
        rvs_data = self.make_cum_posterior_dirichlet(group=group, N_sample=N_sample)
        plt.figure(figsize=(10, 6))
        for i, rvs in enumerate(rvs_data):
            plt.plot(rvs[:, 0], rvs[:, 1], 'o', alpha=0.5, label=f'Alpha {i+1}')
        plt.title(f'Dirichlet Random Variates for Group {group}')
    
    
class BaysMultinomial(MultinomMixin, BayesianModel, PlotManager):
    def __init__(self, prior=None):
        BayesianModel.__init__(self)
        MultinomMixin.__init__(self, prior=prior)



if __name__ == "__main__":

    # generate  random multinomial data

    # Create data type = Bernoulli object
    Multi_test = BaysMultinomial(prior=np.ones(3))

    # create data for two groups
    p_A = [0.1, 0.2,  0.7]
    p_B = [0.2, 0.5, 0.3]
    for n in range(20):
        n_trial = np.random.randint(10,50)
        Multi_test.add_rand_experiment(n_trial, p_A, group="A")
        Multi_test.add_rand_experiment(n_trial, p_B, group="B")

    # Beta marginal posterior for category 0
    p_pts, rvs_data, pdf_data = Multi_test.make_cum_posterior(category_idx=0)

    # Dirichlet rvs
    rvs_data_dirichlet = Multi_test.make_cum_posterior_dirichlet()