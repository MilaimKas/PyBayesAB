
import numpy as np

from scipy.stats import dirichlet, beta
from scipy.stats import multinomial

from PyBayesAB.config import N_SAMPLE, N_PTS

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
    
    def _get_parameters(self, group,  parameters=None, data=None):
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
        alpha = self._get_parameters(group, parameters, data)
        a_i = alpha[category_idx]
        a_rest = alpha.sum() - a_i
        return beta.rvs(a_i, a_rest, size=N_sample)
    
    def make_rvs_dirichlet(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        """
        Generate random variates from the Dirichlet distribution.

        Args:
            parameters (array, optional): alpha array parameter. Defaults to None.
        """
        alpha = self._get_parameters(group, parameters, data)
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
        
        alpha = self._get_parameters(group, parameters, data)
        
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
        if not data or len(data) == 0:
            raise ValueError(f"No data available for group '{group}' to calculate cumulative posterior parameters.")
        # cumulative alpha vector
        cum_alpha = self.prior.copy()
        alphas = np.zeros((len(data) + 1, cum_alpha.shape[0]))
        alphas[0] = cum_alpha
        for i in range(len(data)):
            cum_alpha += data[i]
            alphas[i + 1] = cum_alpha
        return alphas
    
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
    
    def make_posterior_dirichlet(self, group="A", N_sample=N_SAMPLE):
        """
        Generate posterior distribution of the Dirichlet distribution for the specified group.
        """
        alpha = self.post_parameters(group=group)
        return dirichlet.rvs(alpha, size=N_sample)

    def plot_dirichlet_rvs(self, group="A", N_sample=N_SAMPLE):

        fig, ax = plt.subplots()

        if group == "diff":
            rvs_data_A = self.make_posterior_dirichlet(group="A", N_sample=N_sample)
            rvs_data_B = self.make_posterior_dirichlet(group="B", N_sample=N_sample)
            rvs = np.array(rvs_data_A) - np.array(rvs_data_B)
            y_label= "Difference in probabilities (A - B)"
            # plot 0 line
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        
        if group in ["A", "B"]:
            rvs = self.make_posterior_dirichlet(group=group, N_sample=N_sample)
            y_label = f"Group {group} probabilities"
        
        # plot the Dirichlet rvs as lines by taking the max and min of each category
        max_arr =  np.max(rvs, axis=0)
        min_arr =  np.min(rvs, axis=0)  
        ax.fill_between(np.arange(rvs.shape[1]), min_arr, max_arr, alpha=0.5, label=f"Group {group} Dirichlet rvs")
        # plot the mean of the rvs
        mean_arr = np.mean(rvs, axis=0)
        ax.scatter(np.arange(rvs.shape[1]), mean_arr, color='blue', label="mean")
        ax.plot(np.arange(rvs.shape[1]), mean_arr, color='blue', linestyle='--', linewidth=0.5)    
        # add confidence intervals as error bars
        ci_lower, ci_upper = np.percentile(rvs, [5, 95], axis=0)
        ax.errorbar(np.arange(rvs.shape[1]), mean_arr, yerr=[mean_arr - ci_lower, ci_upper - mean_arr], fmt='o', color='blue', label="95% CI", capsize=5)

        # set xticks
        ax.set_xticks(np.arange(rvs.shape[1]))
        ax.set_xticklabels([f"Category {i}" for i in range(rvs.shape[1])])
        
        ax.set_xlabel("Categories")
        ax.set_ylabel(y_label)

        return fig

    def plot_cum_posterior_dirichlet(self, group="A", N_sample=N_SAMPLE, level=95, figsize=None):
        """
        Plot cumulative posterior evolution for all categories as ribbons.
        Each category gets a colored band showing its credible interval narrowing over experiments.

        Args:
            group (str): "A", "B", "AB", or "diff"
            N_sample (int): Number of Dirichlet samples per experiment step.
            level (int): Credible interval percentage (e.g. 95).
            figsize (tuple, optional): Figure size.
        """
        fig, ax = plt.subplots(figsize=figsize)
        tail = (100 - level) / 2

        n_categories = len(self.prior)
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_categories)]

        def _plot_ribbons(ax, rvs_list, colors, linestyle="-", label_suffix=""):
            experiments = np.arange(len(rvs_list))
            for cat_idx in range(n_categories):
                samples = [rvs[:, cat_idx] for rvs in rvs_list]
                means = [np.mean(s) for s in samples]
                ci_low = [np.percentile(s, tail) for s in samples]
                ci_high = [np.percentile(s, 100 - tail) for s in samples]

                ax.fill_between(experiments, ci_low, ci_high, alpha=0.2, color=colors[cat_idx])
                ax.plot(experiments, means, color=colors[cat_idx], linestyle=linestyle,
                        label=f"Category {cat_idx}{label_suffix}")

        if group in ["A", "B"]:
            rvs_list = self.make_cum_posterior_dirichlet(group=group, N_sample=N_sample)
            _plot_ribbons(ax, rvs_list, colors)
            ax.set_ylabel(f"Group {group} probability")

        elif group == "AB":
            self._check_missing_data()
            rvs_A = self.make_cum_posterior_dirichlet(group="A", N_sample=N_sample)
            rvs_B = self.make_cum_posterior_dirichlet(group="B", N_sample=N_sample)
            _plot_ribbons(ax, rvs_A, colors, linestyle="-", label_suffix=" (A)")
            _plot_ribbons(ax, rvs_B, colors, linestyle="--", label_suffix=" (B)")
            ax.set_ylabel("Probability")

        elif group == "diff":
            self._check_missing_data()
            rvs_A = self.make_cum_posterior_dirichlet(group="A", N_sample=N_sample)
            rvs_B = self.make_cum_posterior_dirichlet(group="B", N_sample=N_sample)
            rvs_list = [a - b for a, b in zip(rvs_A, rvs_B)]
            _plot_ribbons(ax, rvs_list, colors)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax.set_ylabel("Difference in probability (A - B)")

        else:
            raise ValueError("group must be 'A', 'B', 'AB', or 'diff'")

        ax.set_xlabel("Experiments")
        ax.legend()
        return fig

class BaysMultinomial(MultinomMixin, BayesianModel, PlotManager):
    def __init__(self, prior=None):
        BayesianModel.__init__(self)
        MultinomMixin.__init__(self, prior=prior)
        self._initialize_cache()

    def __getitem__(self, idx):
        return MultinomialMarginalComponent(self, idx)

    def plot_cum_posterior(self, category_idx=0, group="A", type="2D",
                           N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        """
        Plot cumulative posterior for a specific category's marginal Beta distribution.

        Args:
            category_idx (int): Which category to plot. Defaults to 0.
            group (str): "A", "B", "AB", or "diff"
            type (str): "1D", "2D", or "3D"
        """
        return super().plot_cum_posterior(
            group=group, type=type, N_sample=N_sample,
            para_range=para_range, N_pts=N_pts,
            category_idx=category_idx
        )

class MultinomialMarginalComponent(BayesianModel):
    """
    Marginal component for a specific category of the multinomial distribution.
    This class allows for the calculation of the marginal posterior distribution for a specific category.
    It inherits from BayesianModel and uses the parent model to access the data and parameters.
    """
    def __init__(self, parent_model, category_idx):
        self.parent = parent_model
        self.category_idx = category_idx

    def make_rvs_diff(self, N_sample=N_SAMPLE):
        return self.parent.make_rvs(group="A", category_idx=self.category_idx, N_sample=N_sample) - \
               self.parent.make_rvs(group="B", category_idx=self.category_idx, N_sample=N_sample)

    def make_rvs(self, group="A", N_sample=N_SAMPLE):
        return self.parent.make_rvs(group=group, category_idx=self.category_idx, N_sample=N_sample)

    def make_pdf(self, group="A", **kwargs):
        return self.parent.make_pdf(group=group, category_idx=self.category_idx, **kwargs)




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

    #plot Dirichlet rvs
    fig = Multi_test.plot_dirichlet_rvs(group="diff", N_sample=1000)
    plt.show(block=True)