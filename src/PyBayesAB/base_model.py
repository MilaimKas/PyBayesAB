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
        if group == "A":
            return self.dataA
        elif group == "B":
            return self.dataB
        else:
            raise ValueError("Group must be either 'A' or 'B'")
        
    def add_experiment(self, value, group="A"):
        if group not in ["A", "B"]:
            raise ValueError("Group must be 'A' or 'B'")
        data = self.dataA if group == "A" else self.dataB
        data.append(value)

    def make_rvs_diff(self, N_sample=N_SAMPLE):
        rvs_A = self.make_rvs(group="A", N_sample=N_sample)
        rvs_B = self.make_rvs(group="B", N_sample=N_sample)
        return rvs_A-rvs_B

    def prob_best(self):
        """_summary_

        Args:
            rvs (_type_): _description_

        Returns:
            _type_: _description_
        """
        rvs = self.make_rvs_diff()
        return 100*(np.mean(rvs > 0))

    def hdi(self, group="diff", level=95, post_type="rvs"):
        """_summary_

        Args:
            distribution (_type_): _description_
            level (int, optional): _description_. Defaults to 95.
        """
        if group=="diff" and post_type=="pdf":
            print("Warning: need rvs for hdi of difference")
        if group=="diff":
            post = self.make_rvs_diff()
        elif post_type=="pdf":
            post = self.make_pdf(group=group)
        else:
            post = self.make_rvs(group=group)
        return helper.hdi(post, level=level/100)

    def rope(self, interval, group="diff"):
        """_summary_

        Args:
            rvs (_type_): _description_
            interval (_type_): _description_

        Returns:
            _type_: _description_
        """
        if group == "diff":
            rvs = self.make_rvs_diff()
        else:
            rvs = self.make_rvs(group=group)
        return 1-(np.mean((rvs<np.max(interval)) & (rvs>np.min(interval)))) 

    def rope_decision(self, interval, group="diff", level=95):
        """_summary_

        Args:
            rvs (_type_): _description_
            interval (_type_): _description_
            level (int, optional): _description_. Defaults to 95.
        """
        raise NotImplementedError
        hdi_low, hdi_up = hdi(rvs, level=level)
        rope_low, rope_up = rope(rvs, interval)

        return

    def map_from_rvs(self, method='kde', bins=30):
        """
        Estimates the Maximum A Posteriori (MAP) value from posterior samples.

        Args:
            rvs (array-like): Samples from the posterior distribution.
            method (str): Method to estimate the MAP ('kde' or 'hist').
            bins (int): Number of bins to use if method='hist'.

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
        p_H0 = 1-(np.mean((rvs<np.max(H0)) & (rvs>np.min(H0))))
        H0 = f"Parameter between {np.min(H0):.2f} and {np.max(H0):.2f}"

        if H1 is None:
            H1 = "Parameter larger than 0"
            p_H1 = np.mean(rvs>0)
        else:
            if not isinstance(H1, (list, np.ndarray, tuple)) and len(H1) == 2:
                raise ValueError("Alternative hypothesis must be a interval in values, array or list of length two")
            else:
                H1 = f"Parameter between {np.min(H1):.2f} and {np.max(H1):.2f}"
                p_H1 = 1-(np.mean((rvs<np.max(H1)) & (rvs>np.min(H1))))

        BF = p_H1/p_H0

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

