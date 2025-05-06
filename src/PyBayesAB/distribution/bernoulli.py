

"""
Bayesian A/B testing for Bernoulli distributed data.

This module provides classes for performing Bayesian A/B testing when the
underlying data is assumed to follow a Bernoulli distribution. This is
common in scenarios like conversion rates (e.g., click-through rates,
sign-up rates), where each observation is a binary outcome (success/failure,
1/0).

The core idea is to model the probability of success (p) for each group (A and B)
using a Beta distribution as the prior. After observing data (number of successes
and failures), the posterior distribution for p is also a Beta distribution.
This is due to the conjugacy between the Bernoulli likelihood and the Beta prior.

Classes:
    BernoulliMixin: A mixin class providing the core Bayesian logic for
                    Bernoulli-distributed data with a Beta prior.
    BaysBernoulli:  A class that combines `BernoulliMixin` with `BayesianModel`
                    and `PlotManager` to provide a full A/B testing framework
                    for Bernoulli data, including data handling and plotting.

Constants:
    PRIOR_TYPES: A dictionary of common prior parameters (alpha, beta) for the
                 Beta distribution.
    PARA_RANGE: The typical parameter range for the Bernoulli probability (0 to 1).
"""
import numpy as np

from scipy.stats import beta
from scipy.stats import bernoulli

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  


from PyBayesAB import N_SAMPLE, N_PTS

PRIOR_TYPES = {
    'Bayes-Laplace': [1, 1],
    'Jeffreys': [0.5, 0.5],
    'Neutral': [1/3, 1/3],
    'Haldane': [1e-3, 1e-3]
}

PARA_RANGE = [0, 1]

class BernoulliMixin:
    """
    A mixin class providing core Bayesian A/B testing logic for Bernoulli distributed data.

    This class handles the mathematical operations for updating prior beliefs
    (Beta distribution) with observed data (successes and failures) to obtain
    posterior distributions for the Bernoulli probability parameter.

    It assumes a Bernoulli likelihood for the data and a Beta distribution for
    both the prior and posterior of the Bernoulli probability parameter `p`.

    Attributes:
        prior (list or tuple): A list or tuple `[alpha, beta]` representing the
            parameters of the Beta prior distribution.
        parameter_name (str): The name of the parameter being estimated,
            initialized to "Bernoulli probability".
    """
    def __init__(self, prior):
        """
        Initialize the BernoulliMixin with prior parameters.

        Args:
            prior (list or tuple): A list or tuple `[alpha, beta]` representing the
                parameters of the Beta prior distribution. Alpha and beta must be
                positive.
        """
        if not (isinstance(prior, (list, tuple)) and len(prior) == 2 and
                all(isinstance(p, (int, float)) and p > 0 for p in prior)):
            raise ValueError("Prior must be a list or tuple of two positive numbers [alpha, beta].")

        self.prior = prior
        self.parameter_name = "Bernoulli probability"

    def get_parameters(self, parameters, group, data):
        """
        Retrieves the Beta distribution parameters (alpha, beta).

        If `parameters` are provided, they are used. Otherwise, posterior
        parameters are calculated based on the data for the specified `group`.

        Args:
            parameters (list or tuple, optional): A list or tuple `[alpha, beta]`
                of Beta parameters. If None, parameters are computed from data.
            group (str): The experimental group ("A", "B", etc.) from which to
                derive parameters if `parameters` is None.
            data (array-like, optional): The data to use for calculating posterior
                parameters. If None, `self.return_data(group)` is used.
                Expected format is an array where each row is `[successes, failures]`.

        Returns:
            tuple: A tuple `(alpha, beta)` of Beta distribution parameters.

        Raises:
            ValueError: If `parameters` is provided but not of length 2.
        """
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Beta posterior needs 2 parameters: alpha and beta")
            else:
                a,b = parameters
        else:
            a,b = self.post_parameters(group=group, data=data)
        return a, b

    def add_rand_experiment(self, n, p, group="A"):
        """
        Simulates and adds a random experiment's results.

        Generates `n` random samples from a Bernoulli distribution with
        probability `p`, counts the number of successes (hits) and failures,
        and adds this `[hits, fails]` pair to the specified group's data.

        Args:
            n (int): The number of trials in the simulated experiment.
            p (float): The probability of success for each trial (0 <= p <= 1).
            group (str, optional): The experimental group to which the results
                are added. Defaults to "A".
        """
        hits = np.sum(bernoulli.rvs(p, size=n))
        fails = n - hits
        self.add_experiment([hits, fails], group=group)

    def make_rvs(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        """
        Generates random variates (samples) from the Beta posterior distribution.

        Args:
            parameters (list or tuple, optional): `[alpha, beta]` parameters for the
                Beta distribution. If None, they are derived from `data` or
                the stored data for `group`.
            data (array-like, optional): Data to compute posterior parameters if
                `parameters` is None.
            group (str, optional): Group to use if `parameters` and `data` are None.
                Defaults to "A".
            N_sample (int, optional): The number of random variates to generate.
                Defaults to `N_SAMPLE` from `PyBayesAB` config.

        Returns:
            numpy.ndarray: An array of `N_sample` random variates from the
                           Beta posterior.
        """
        a,b = self.get_parameters(parameters, group, data)
        return beta.rvs(a,b,size=N_sample)

    def make_pdf(self, parameters=None, data=None, group="A", p_pts=None, para_range=PARA_RANGE):
        """
        Calculates the probability density function (PDF) of the Beta posterior.

        Args:
            parameters (list or tuple, optional): `[alpha, beta]` parameters for the
                Beta distribution. If None, derived from `data` or stored data.
            data (array-like, optional): Data to compute posterior parameters if
                `parameters` is None.
            group (str, optional): Group to use if `parameters` and `data` are None.
                Defaults to "A".
            p_pts (numpy.ndarray, optional): Array of points at which to evaluate
                the PDF. If None, a default range is generated using `para_range`
                and `N_PTS`.
            para_range (list or tuple, optional): `[min_p, max_p]` for generating
                `p_pts` if it's None. Defaults to `PARA_RANGE`.

        Returns:
            numpy.ndarray: The PDF values evaluated at `p_pts`.
        """
        a,b = self.get_parameters(parameters, group, data)
        if p_pts is None:
            p_pts = np.linspace(para_range[0], para_range[1], N_PTS)
        return beta.pdf(p_pts,a,b)

    def post_parameters(self, data=None, group="A"):
        """
        Calculates the parameters (alpha, beta) of the Beta posterior distribution.

        The posterior alpha is the sum of prior alpha and total successes.
        The posterior beta is the sum of prior beta and total failures.

        Args:
            data (array-like, optional): The observed data, where each row is
                `[successes, failures]`. If None, data for the specified `group`
                is fetched using `self.return_data(group)`.
            group (str, optional): The experimental group for which to calculate
                posterior parameters if `data` is None. Defaults to "A".

        Returns:
            tuple: A tuple `(posterior_alpha, posterior_beta)`.
        """
        if data is None:
            data = np.array(self.return_data(group))
        a = np.sum(data[:, 0]) + self.prior[0]
        b = np.sum(data[:, 1]) + self.prior[1]
        return a,b

    def make_cum_post_para(self, group="A"):
        """
        Calculates cumulative posterior parameters (alpha, beta) for each experiment.

        This is useful for observing how the posterior distribution evolves as
        more data (experiments) are added. The prior is added to the first
        experiment's data.

        Args:
            group (str, optional): The experimental group for which to calculate
                cumulative parameters. Defaults to "A".

        Returns:
            tuple: A tuple `(cumulative_alphas, cumulative_betas)`, where each
                   is a NumPy array of the cumulative parameters after each
                   experiment.
        """
        data =  np.array(self.return_data(group))
        if data.shape[0] == 0: # Handle case with no data
            return np.array([self.prior[0]]), np.array([self.prior[1]])

        # Make a copy to avoid modifying the original data stored in the object
        data_copy = data.copy()
        # add prior to the first data point's contribution
        data_copy[0,0] += self.prior[0]
        data_copy[0,1] += self.prior[1]
        # cumulative hits and fails
        cumsum_alpha = np.cumsum(data_copy[:,0])
        cumsum_beta = np.cumsum(data_copy[:,1])
        return cumsum_alpha,cumsum_beta

    def post_pred(self, size=1, group="A"):
        """
        Generates samples from the posterior predictive distribution.

        The posterior predictive distribution for a Bernoulli likelihood with a
        Beta prior gives the probability of success for a new trial, given the
        observed data. It's equivalent to drawing from a Bernoulli distribution
        where the success probability `p_new = posterior_alpha / (posterior_alpha + posterior_beta)`.

        Args:
            size (int, optional): The number of samples to draw from the
                posterior predictive distribution. Defaults to 1.
            group (str, optional): The experimental group for which to calculate
                the posterior predictive. Defaults to "A".

        Returns:
            numpy.ndarray: An array of `size` samples (0s or 1s) from the
                           posterior predictive distribution.
        """
        a,b = self.post_parameters(group=group)
        p_new = a/(a+b)
        return bernoulli.rvs(p_new, size=size)

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        """
        Generates data for plotting cumulative posterior distributions.

        For each experiment added sequentially, it calculates the posterior
        parameters and then generates random variates (RVS) and the PDF
        for that cumulative posterior.

        Args:
            group (str, optional): The experimental group. Defaults to "A".
            N_sample (int, optional): Number of random variates for each
                cumulative posterior. Defaults to `N_SAMPLE`.
            para_range (list or tuple, optional): `[min_p, max_p]` for PDF
                calculation. If None, `PARA_RANGE` is used.
            N_pts (int, optional): Number of points for PDF calculation.
                Defaults to `N_PTS`.

        Returns:
            tuple: `(p_pts, rvs_data, pdf_data)`
                - `p_pts` (numpy.ndarray): Points at which PDFs are evaluated.
                - `rvs_data` (list): List of NumPy arrays, each containing RVS
                  for a cumulative posterior.
                - `pdf_data` (list): List of NumPy arrays, each containing PDF values
                  for a cumulative posterior.
        """
        if para_range is None:
            para_range=PARA_RANGE
        # create list of rvs and pdf
        a_cum, b_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        p_pts = np.linspace(para_range[0], para_range[1], N_pts)
        for a,b in zip(a_cum, b_cum):
            rvs_data.append(self.make_rvs(parameters=[a,b], N_sample=N_sample))
            pdf_data.append(self.make_pdf(parameters=[a,b], p_pts=p_pts))
        return p_pts, rvs_data, pdf_data


class BaysBernoulli(BernoulliMixin, BayesianModel, PlotManager):
    """
    Bayesian A/B testing model for Bernoulli distributed data.

    This class integrates the Bernoulli likelihood and Beta prior/posterior logic
    from `BernoulliMixin` with the general data handling capabilities of
    `BayesianModel` and plotting functionalities from `PlotManager`.

    It allows users to define experiments, add data (successes and failures),
    and then analyze the results using Bayesian inference, including calculating
    posterior distributions, probabilities of one group being better than another,
    credible intervals, and generating various plots.

    Example:
        >>> model = BaysBernoulli(prior_type='Jeffreys')
        >>> # Group A: 100 successes (hits), 400 failures (out of 500 trials)
        >>> model.add_experiment([100, 400], group="A")
        >>> # Group B: 120 successes, 380 failures (out of 500 trials)
        >>> model.add_experiment([120, 380], group="B")
        >>>
        >>> # Get posterior parameters for group A
        >>> alpha_A, beta_A = model.post_parameters(group="A")
        >>> print(f"Group A posterior: Beta({alpha_A:.2f}, {beta_A:.2f})")
        >>>
        >>> # Plot posterior distributions
        >>> model.plot_final_posterior(group="A")
        >>> model.plot_final_posterior(group="B")
        >>> model.plot_final_posterior(group="diff") # Difference A-B
        >>>
        >>> # Probability that B is better than A
        >>> prob_B_better_A = model.prob_best()["B"]
        >>> print(f"Probability B is better than A: {prob_B_better_A:.2%}")
    """
    def __init__(self, prior_type="Bayes-Laplace"):
        """
        Initializes the BaysBernoulli model.

        Args:
            prior_type (str, optional): The type of Beta prior to use.
                Must be a key in `PRIOR_TYPES`. Defaults to "Bayes-Laplace".
                Available types:
                - 'Bayes-Laplace': Beta(1, 1) - Uniform prior.
                - 'Jeffreys': Beta(0.5, 0.5) - Jeffreys prior.
                - 'Neutral': Beta(1/3, 1/3) - A less informative prior.
                - 'Haldane': Beta(1e-3, 1e-3) - Approximates log-uniform,
                                               can be problematic if no successes
                                               or no failures are observed.
        Raises:
            KeyError: If `prior_type` is not found in `PRIOR_TYPES`.
        """
        if prior_type not in PRIOR_TYPES:
            raise KeyError(f"Unknown prior_type: '{prior_type}'. "
                           f"Available types are: {list(PRIOR_TYPES.keys())}")
        BayesianModel.__init__(self)
        BernoulliMixin.__init__(self, prior=PRIOR_TYPES[prior_type])



    
