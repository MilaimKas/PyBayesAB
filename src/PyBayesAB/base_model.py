import numpy as np

from PyBayesAB import helper, bayesian_functions
from PyBayesAB import N_SAMPLE

import  pandas as pd

class BayesianModel:

    def __init__(self):

        self.dataA = []
        self.dataB = []        

        # cahche for posterior samples and pdfs
    
    def add_test_result(self, df:pd.DataFrame):
        """
        Store the result of the test from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the test results with columns 'group', 'values', and 'experiment' or 'date'
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if 'group' not in df.columns or 'values' not in df.columns:
            raise ValueError("DataFrame must contain 'group' and 'values' columns")
        
        if 'experiment' not in df.columns and 'date' not in df.columns:
            raise ValueError("DataFrame must contain either 'experiment' or 'date' column")
        else:
            experiment_col = 'experiment' if 'experiment' in df.columns else 'date'
        
        df.groupby([experiment_col, 'group'], sort=True)['values'].apply(list).reset_index(name='values').apply(
            lambda row: self.add_experiment(values=row['values'], group=row['group']), axis=1)

        # initialize cache
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize the cache for posterior samples."""
        self.rvs_A = None
        self.rvs_B = None   
        self.rvs_cum_A = None   
        self.rvs_cum_B = None   
    
    def __mul__(self, other):
        return NotImplementedError

    def return_data(self, group):
        if group == "A":
            return self.dataA
        elif group == "B":
            return self.dataB
        else:
            raise ValueError("Group must be either 'A' or 'B'")
    
    def add_experiment(self, values, group="A"):
        if group not in ["A", "B"]:
            raise ValueError("Group must be 'A' or 'B'")
        if not isinstance(values, (list, np.ndarray)):
            values = [values]
        data = self.dataA if group == "A" else self.dataB
        data.append(values)
        # re-initialize cache to ensure new data is considered in posterior calculations
        self._initialize_cache()

    def _check_missing_data(self):
        if len(self.dataA) != len(self.dataB):  
            print("WARNING: Data for groups A and B do not have the same number of experiments. Considering missing data as zero.")
            # fill missing data with zeros
            max_length = max(len(self.dataA), len(self.dataB))
            self.dataA += [[]]*(max_length - len(self.dataA))
            self.dataB += [[]]*(max_length - len(self.dataB))

    def make_rvs_diff(self, N_sample=N_SAMPLE, post_kwargs={}):
        if self.rvs_A is None or self.rvs_B is None:
            #  check if dataA and dataB are same length
            self._check_missing_data()
            # make rvs for both groups
            self.rvs_A = self.make_rvs(group="A", N_sample=N_sample, **post_kwargs)
            self.rvs_B = self.make_rvs(group="B", N_sample=N_sample, **post_kwargs)
        return self.rvs_A-self.rvs_B

    def prob_best(self, post_kwargs={}):
        """
        Get the probability that group A is better than group B.

        Args:
            rvs (array): Samples from the posterior distribution.

        Returns:
            float: Probability in percentage that group A is better than group B.
        """
        rvs = self.make_rvs_diff(**post_kwargs)
        return 100*(np.mean(rvs > 0))

    def hdi(self, group="diff", level=95, post_type="rvs", norm_app=False, post_kwargs={}):
        """
        Calculate the Highest Density Interval (HDI) for the posterior distribution.

        Args:
            distribution (array or callable): Samples from the posterior distribution or a callable pdf function.
            level (int, optional): level in percentage. Defaults to 95.
        """
        if group=="diff" and post_type=="pdf":
            print("Warning: need rvs for hdi of difference. Will proceed using rvs.")
        if group=="diff":
            post = self.make_rvs_diff(**post_kwargs)
        elif post_type=="pdf":
            post = self.make_pdf(group=group, **post_kwargs)
        else:
            post = self.make_rvs(group=group, **post_kwargs)
        return helper.hdi(post, level=level/100, norm_app=norm_app)

    def rope(self, interval, group="diff", post_kwargs={}):
        """
        Calculate the probability that the difference in parameters is within the ROPE (Region of Practical Equivalence).

        Args:
            rvs (array): Samples from the posterior distribution.   
            interval (list): ROPE interval, e.g. [-1, 1] of the parameters.

        Returns:
            float: Probability in percentage that the difference is within the ROPE interval.
        """
        if group == "diff":
            rvs = self.make_rvs_diff(**post_kwargs)
        else:
            rvs = self.make_rvs(group=group, **post_kwargs)
        return bayesian_functions.rope(rvs=rvs, interval=interval)*100

    def rope_decision(self, rope_interval, level=95, post_kwargs={}):
        """
        Make a decision based on the ROPE (Region of Practical Equivalence) interval.

        Args:
            rvs (array): Samples from the posterior distribution.
            rope_interval (list): ROPE interval, e.g. [-1, 1] of the parameters.
            level (int, optional): Level in percentage. Defaults to 95.
        """
        return bayesian_functions.rope_decision(rvs=self.make_rvs_diff(**post_kwargs), rope_interval=rope_interval, level=level)

    def map(self, method='median', post_kwargs={}):
        """
        Estimates the Maximum A Posteriori (MAP) value from posterior samples.

        Args:
            rvs (array-like): Samples from the posterior distribution.
            method (str): Method to estimate the MAP ('kde' or 'median').

        Returns:
            float: MAP estimate based on the provided samples.
        """
        rvs = self.make_rvs_diff(**post_kwargs)
        
        return bayesian_functions.map(rvs, method=method)

    def bayesian_factor(self, H1=None, H0=None, prior=None, scale_factor=0.1, post_kwargs={}):
        """
        Calculate the Bayes factor for the null hypothesis (H0) and alternative hypothesis (H1).
        Args:
            H1 (list, optional): Alternative hypothesis interval, e.g. [-1, 1]. Defaults to None.
            H0 (list, optional): Null hypothesis interval, e.g. [-1, 1]. Defaults to None.
            prior (list, optional): Prior distribution parameters. Defaults to None.        
            scale_factor (float, optional): Factor to scale the interquartile range (IQR) for H0. Defaults to 0.1.
        Returns:
            str: A string summarizing the Bayes factor and the evidence it provides for H0 and H1.
        """
        rvs = self.make_rvs_diff(**post_kwargs)

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
    
    def summary_result(self, rope_interval, level, post_kwargs={}):
        """
        Generate a summary of the Bayesian metrics.

        Args:
            rope_interval (list): ROPE interval, e.g. [-1, 1] of the parameters.
            level (float): Confidence level for the HDI, e.g. 95.
            post_kwargs (dict, optional): Additional args for the posterior calculation function. Defaults to {}.

        Returns:
            str
        """

        result = "Bayesian metrics summary: \n\n"

        result += f"Probablity that A is better than B = {self.prob_best(post_kwargs=post_kwargs):.2f}% \n\n"

        hdi = self.hdi(level=level, post_kwargs=post_kwargs)
        result += f"There is 95% that the difference in {self.parameter_name} is between {hdi[0]:.2f} and {hdi[1]:.2f} \n\n"

        result += f"The MAP (maximum a posterior estimate) if {self.map():.2f} \n\n"

        result += f"Probability that the difference is within the ROPE (region of practical equivalence) is {self.rope(interval=rope_interval, post_kwargs=post_kwargs):.1f}% \n\n"

        result += f"ROPE-based decision: {self.rope_decision(rope_interval, level=level, post_kwargs=post_kwargs)}  \n\n"

        result += "Bayes factor (A vs B vs null): \n"
        result += self.bayesian_factor(H0=rope_interval, post_kwargs=post_kwargs) + "\n\n"

        return result
    
    def __add__(self, other):
        if not isinstance(other, BayesianModel):
            raise ValueError("Can only add another BayesianModel instance")
        
        rvs_A = self.make_rvs(group="A")
        rvs_B = other.make_rvs(group="B")
        rvs_A_other = other.make_rvs(group="A")
        rvs_B_self = self.make_rvs(group="B")

        rvs_A_add = rvs_A + rvs_A_other
        rvs_B_add = rvs_B + rvs_B_self

        raise NotImplementedError("Addition of BayesianModel instances is not implemented yet")

    def __div__(self, other):      
        raise NotImplementedError("Division of BayesianModel instances is not implemented yet")
    
    def __mul__(self, other):
        raise NotImplementedError("Multiplication of BayesianModel instances is not implemented yet")
    
    def __sub__(self, other):
        raise NotImplementedError("Subtraction of BayesianModel instances is not implemented yet")

if __name__ == "__main__":

    model = BayesianModel()

    # from data frame
    data = pd.DataFrame({
        'group': ['A', 'B', 'A', 'B', 'B'],
        'values': [5, 7, 6, 8, 14],
        'experiment': [1, 1, 2, 2, 1]
    })  
    model.add_test_result(data)
    
    model.add_experiment(values=[21, 9], group="A")
    model.add_experiment(values=[75], group="A")   

    print(model._check_missing_data())

    print(model.dataA)
    print(model.dataB) 