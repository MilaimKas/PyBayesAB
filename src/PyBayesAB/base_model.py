import numpy as np
from PyBayesAB import helper, plot_functions, bayesian_functions
from PyBayesAB import N_SAMPLE

class BayesianModel:

    def __init__(self):

        self.dataA = []
        self.dataB = []
    
    """
    Data process and plotting 
    """
        
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

    def MAP(self, group="diff"):
        raise NotImplementedError
        return

    def bayesian_factor(self, H1=None, H0=None, prior=None):
        
        rvs = self.make_rvs_diff()

        if H1 is None:
            p_H1 = np.sum(rvs)
        else:
            if not isinstance(H1, (list, np.array, tuple)):
                raise ValueError("Alternative hypothesis must be a interval in values, array or list of length two")
            p_H1 = self.rope(rvs, H1)

        if H0 is None:
            p_H0 = 1/len(rvs)
        else:
            if not isinstance(H0, (list, np.array, tuple)):
                raise ValueError("Alternative hypothesis must be a interval in values, array or list of length two")
            p_H0 = self.rope(rvs, H0)
        
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

        return "Bayes factor is {:.2f}, thus providing ".format(BF) + text

