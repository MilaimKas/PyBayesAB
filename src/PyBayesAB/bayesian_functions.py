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
    return 100*(np.mean((rvs<max(interval)) & rvs > min(interval))) 

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

def bayesian_factor():
    raise NotImplementedError
    return