import numpy as np

from PyBayesAB import helper, plot_functions, bayesian_functions
from PyBayesAB import N_SAMPLE

from scipy.stats import gaussian_kde

class BayesianModel:

    def __init__(self):
        self.dataA = []
        self.dataB = []
    
    """
    Data process and plotting 
    """
    
    def return_data(self, group):
        """
        Returns the data for the specified group.

        Args:
            group (str): The group to return data for ("A" or "B").

        Returns:
            list: The data for the specified group.
        """
        if group == "A":
            return self.dataA
        elif group == "B":
            return self.dataB
        else:
            raise ValueError("Group must be either 'A' or 'B'")
        
    def add_experiment(self, value, group="A"):
        """
        Adds an experiment value to the specified group.

        Args:
            value (float): The value to add.
            group (str, optional): The group to add the value to ("A" or "B"). Defaults to "A".
        """
        if group not in ["A", "B"]:
            raise ValueError("Group must be 'A' or 'B'")
        data = self.dataA if group == "A" else self.dataB
        data.append(value)

    def make_rvs_diff(self, N_sample=N_SAMPLE):
        """
        Generates random value samples from the posterior distribution for both groups and returns their difference.

        Args:
            N_sample (int, optional): Number of samples to generate. Defaults to N_SAMPLE.

        Returns:
            np.array: The difference between the samples from group A and group B.
        """
        rvs_A = self.make_rvs(group="A", N_sample=N_sample)
        rvs_B = self.make_rvs(group="B", N_sample=N_sample)
        return rvs_A - rvs_B

    def prob_best(self):
        """
        Calculates the probability that group A is better than group B.

        Returns:
            float: The probability that group A is better than group B.
        """
        rvs = self.make_rvs_diff()
        return 100 * np.mean(rvs > 0)

    def hdi(self, group="diff", level=95, post_type="rvs"):
        """
        Calculates the highest density interval (HDI) for the specified group.

        Args:
            group (str, optional): The group to calculate the HDI for ("A", "B", or "diff"). Defaults to "diff".
            level (int, optional): The confidence level for the HDI. Defaults to 95.
            post_type (str, optional): The type of posterior distribution ("rvs" or "pdf"). Defaults to "rvs".

        Returns:
            tuple: The lower and upper bounds of the HDI.
        """
        if group == "diff" and post_type == "pdf":
            print("Warning: need rvs for hdi of difference. Will proceed using rvs.")
        if group == "diff":
            post = self.make_rvs_diff()
        elif post_type == "pdf":
            post = self.make_pdf(group=group)
        else:
            post = self.make_rvs(group=group)
        return helper.hdi(post, level=level / 100)

    def rope(self, interval, group="diff"):
        """
        Calculates the region of practical equivalence (ROPE) for the specified group.

        Args:
            interval (list or tuple): The interval for the ROPE.
            group (str, optional): The group to calculate the ROPE for ("A", "B", or "diff"). Defaults to "diff".

        Returns:
            float: The percentage of the posterior distribution within the ROPE.
        """
        if group == "diff":
            rvs = self.make_rvs_diff()
        else:
            rvs = self.make_rvs(group=group)
        return 100 * (1 - np.mean((rvs < np.max(interval)) & (rvs > np.min(interval))))

    def rope_decision(self, interval, group="diff", level=95):
        """
        Makes a decision based on the ROPE and HDI for the specified group.

        Args:
            interval (list or tuple): The interval for the ROPE.
            group (str, optional): The group to make the decision for ("A", "B", or "diff"). Defaults to "diff".
            level (int, optional): The confidence level for the HDI. Defaults to 95.
        """
        raise NotImplementedError

    def map_from_rvs(self, method='kde', bins=30):
        """
        Estimates the Maximum A Posteriori (MAP) value from posterior samples.

        Args:
            method (str, optional): Method to estimate the MAP ('kde' or 'hist'). Defaults to 'kde'.
            bins (int, optional): Number of bins to use if method='hist'. Defaults to 30.

        Returns:
            float: MAP estimate based on the provided samples.
        """
        rvs = self.make_rvs_diff()

        if method == 'kde':
            # Kernel Density Estimation (KDE)
            kde = gaussian_kde(rvs)
            x_vals = np.linspace(np.min(rvs), np.max(rvs), 1000)
            densities = kde(x_vals)
            map_estimate = x_vals[np.argmax(densities)]
        elif method == 'hist':
            # Histogram-based estimation
            hist, edges = np.histogram(rvs, bins=bins, density=True)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            map_estimate = bin_centers[np.argmax(hist)]
        else:
            raise ValueError("Invalid method. Choose 'kde' or 'hist'.")
        
        return map_estimate

    def bayesian_factor(self, H1=None, H0=None, prior=None, scale_factor=0.1):
        """
        Calculates the Bayes factor for the specified hypotheses.

        Args:
            H1 (list or tuple, optional): The interval for the alternative hypothesis. Defaults to None.
            H0 (list or tuple, optional): The interval for the null hypothesis. Defaults to None.
            prior (list or tuple, optional): The prior distribution. Defaults to None.
            scale_factor (float, optional): The scale factor for the null hypothesis interval. Defaults to 0.1.

        Returns:
            str: A summary of the Bayes factor and the evidence it provides.
        """
        rvs = self.make_rvs_diff()

        if H0 is None:
            # Use interquartile range (IQR) to dynamically set the scale
            iqr = np.percentile(rvs, 75) - np.percentile(rvs, 25)
            center = 0  # Default center for the null hypothesis
            half_width = scale_factor * iqr if iqr > 0 else scale_factor * np.std(rvs)
            H0 = [center - half_width, center + half_width]
        else:
            if not isinstance(H0, (list, np.array, tuple)):
                raise ValueError("Null hypothesis must be a interval in values, array or list of length two")
        p_H0 = 1 - np.mean((rvs < np.max(H0)) & (rvs > np.min(H0)))
        H0 = f"Parameter between {np.min(H0):.2f} and {np.max(H0):.2f}"

        if H1 is None:
            H1 = "Parameter larger than 0"
            p_H1 = np.mean(rvs > 0)
        else:
            if not isinstance(H1, (list, np.ndarray, tuple)) and len(H1) == 2:
                raise ValueError("Alternative hypothesis must be a interval in values, array or list of length two")
            else:
                H1 = f"Parameter between {np.min(H1):.2f} and {np.max(H1):.2f}"
                p_H1 = 1 - np.mean((rvs < np.max(H1)) & (rvs > np.min(H1)))

        BF = p_H1 / p_H0

        # calculate bayes factor given H0 and H1
        # return plain text 
        text = " "
        if BF < 1: 
            text = "supports for the null hypothesis"
        elif 1 < BF < 3:
            text ="anecdotal evidence for the alternative"
        elif 3 < BF < 10: 
            text = "moderate evidence for the alternative"
        elif 10 < BF < 30: 
            text = "strong evidence for the alternative"
        elif 30 < BF < 100:
            text = "very strong evidence for the alternative"
        else: 
            text = "decisive/extreme evidence for the alternative"

        return f"""
                For the null hypothesis: {H0}
                For the alternative hypothesis: {H1}
                The Bayes factor is {BF:.2f}, thus providing {text}
                """

