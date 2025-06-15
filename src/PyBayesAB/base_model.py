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

    def hdi(self, group="diff", level=95, post_type="rvs", norm_app=False):
        """_summary_

        Args:
            distribution (_type_): _description_
            level (int, optional): _description_. Defaults to 95.
        """
        if group=="diff" and post_type=="pdf":
            print("Warning: need rvs for hdi of difference. Will proceed using rvs.")
        if group=="diff":
            post = self.make_rvs_diff()
        elif post_type=="pdf":
            post = self.make_pdf(group=group)
        else:
            post = self.make_rvs(group=group)
        return helper.hdi(post, level=level/100, norm_app=norm_app)

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
        return bayesian_functions.rope(rvs=rvs, interval=interval)*100

    def rope_decision(self, rope_interval, level=95):
        """_summary_

        Args:
            rvs (_type_): _description_
            rope_interval (_type_): _description_
            level (int, optional): _description_. Defaults to 95.
        """
        return bayesian_functions.rope_decision(rvs=self.make_rvs_diff(), rope_interval=rope_interval, level=level)

    def map(self, method='median'):
        """
        Estimates the Maximum A Posteriori (MAP) value from posterior samples.

        Args:
            rvs (array-like): Samples from the posterior distribution.
            method (str): Method to estimate the MAP ('kde' or 'median').

        Returns:
            float: MAP estimate based on the provided samples.
        """
        rvs = self.make_rvs_diff()
        
        return bayesian_functions.map(rvs, method=method)

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
            H1 = "Parameter larger than 0 or smaller than 0"
            p_H1 = max(np.mean(rvs>0), np.mean(rvs<0))
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
    
    def summary_result(self, rope_interval, level):

        result = ""
        result += f"Probablity that A is better than B = {self.prob_best():.2f}% \n\n"

        hdi = self.hdi(level=level)
        result += f"There is 95% that the difference in Bernoulli probability is between {hdi[0]:.2f} and {hdi[1]:.2f} \n\n"

        result += f"The MAP (maximum a posterior estimate) if {self.map():.2f} \n\n"

        result += f"Probability that the difference is within the ROPE (region of practical equivalence) is {self.rope(interval=rope_interval):.1f}% \n\n"

        result += f"ROPE-based decision: {self.rope_decision(rope_interval, level=level)}  \n\n"

        result += "Bayes factor (A vs B vs null): \n"
        result += self.bayesian_factor() + "\n\n"

        return result

