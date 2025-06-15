"""
 Functions to calculate several Bayesian metrics from the posterior distribution 
"""

import numpy as np
from PyBayesAB import helper
from scipy.stats import gaussian_kde

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
        distribution (array or callable): rvs samples or callable pdf.
        level (int, optional): mass density for high density interval. Defaults to 95%.
    """
    return helper.hdi(distribution, level=level/100)

def rope(rvs, interval):
    """
    Proportion of random samples that fall within the specified interval (ROPE).

    Args:
        rvs (array): Random samples from the posterior distribution.
        interval (list or tuple): The interval for the ROPE.

    Returns:
        float: Proportion of samples within the ROPE interval.
    """
    return (np.mean((rvs<np.max(interval)) & (rvs>np.min(interval)))) 

def rope_decision(rvs, rope_interval, level=95):
    """
    Makes a decision based on the ROPE and HDI for the posterior distribution.

    Args:
        rvs (np.array): Random samples from the posterior distribution.
        interval (list or tuple): The interval for the ROPE.
        level (int, optional): The confidence level for the HDI. Defaults to 95.

    Returns:
        str: Decision about the significance of the test.
    """
    # Calculate HDI
    hdi_low, hdi_up = hdi(rvs, level=level)

    # Check if HDI is > 0  and entirely outside the ROPE
    if hdi_up < 0:
        if hdi_up < np.min(rope_interval):
            return "Practically Significant: Group A is better"
        else:
            return "Statistically Significant: Group A is better"
    elif hdi_low > 0:
        if  hdi_low > np.max(rope_interval):
            return "Practically Significant: Group B is better"
        else:
            return "Statistically Significant: Group B is better"

    # Check if HDI is entirely within the ROPE
    elif hdi_low >= np.min(rope_interval) and hdi_up <= np.max(rope_interval):
        return "Significant: no difference between A and B (within ROPE)"

    # Otherwise, the result is inconclusive
    else:
        return "Inconclusive: needs more data (overlaps with ROPE)"

def bayesian_factor(posterior, H1=None, H0=None, prior=None):

    if H1 is None:
        p_H1 = np.sum(posterior)
    else:
        if not isinstance(H1, (list, np.array, tuple)):
            raise ValueError("Alternative hypothesis must be a interval in values, array or list of length two")
        p_H1 = rope(posterior, H1)

    if H0 is None:
        p_H0 = 1/len(posterior)
    else:
       if not isinstance(H0, (list, np.array, tuple)):
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

def map(rvs, method="median"):
    """
    Estimates the Maximum A Posteriori (MAP) value from posterior samples.

    Args:
        rvs (np.array): Random samples from the posterior distribution.
        method (str, optional): Method to estimate the MAP ("median" or "kde"). Defaults to "median".

    Returns:
        float: MAP estimate based on the provided samples.
    """
    if method == "median":
        # Use the median of the posterior samples as the MAP estimate
        map_estimate = np.median(rvs)
    elif method == "kde":
        # Kernel Density Estimation (KDE)
        kde = gaussian_kde(rvs)
        x_vals = np.linspace(np.min(rvs), np.max(rvs), 1000)
        densities = kde(x_vals)
        map_estimate = x_vals[np.argmax(densities)]
    else:
        raise ValueError("Invalid method. Choose 'median' or 'kde'.")

    return map_estimate