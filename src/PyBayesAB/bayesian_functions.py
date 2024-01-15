"""
 Functions to calculate several Bayesian metrics from the posterior distribution 
"""

import numpy as np
from PyBayesAB import helper

def prob_best(rvs):
    """_summary_

    Args:
        rvs (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 100*(np.mean(rvs > 0))

def hdi(distribution, level=95):
    """_summary_

    Args:
        distribution (_type_): _description_
        level (int, optional): _description_. Defaults to 95.
    """
    return helper.hdi(distribution, level=level/100)

def rope(rvs, interval):
    """_summary_

    Args:
        rvs (_type_): _description_
        interval (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1-(np.mean((rvs<np.max(interval)) & (rvs>np.min(interval)))) 

def rope_decision(rvs, interval, level=95):
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

def MAP():
    raise NotImplementedError
    return

def bayesian_factor(posterior, H1=None, H0=None, prior=None):

    if H1 is None:
        p_H1 = np.sum(posterior)
    else:
        if not isinstance(H1, (list, ndarray, tuple)):
            raise ValueError("Alternative hypothesis must be a interval in values, array or list of length two")
        p_H1 = rope(posterior, H1)

    if H0 is None:
        p_H0 = 1/len(posterior)
    else:
       if not isinstance(H0, (list, ndarray, tuple)):
            raise ValueError("Alternative hypothesis must be a interval in values, array or list of length two")
       p_H0 = rope(posterior, H0)
    
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