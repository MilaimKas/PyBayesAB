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
            return "Group A is better -> Practically Significant"
        else:
            return "Group A is better -> Statistically Significant"
    elif hdi_low > 0:
        if  hdi_low > np.max(rope_interval):
            return "Group B is better -> Practically Significant"
        else:
            return "Group B is better -> Statistically Significant"

    # Check if HDI is entirely within the ROPE
    elif hdi_low >= np.min(rope_interval) and hdi_up <= np.max(rope_interval):
        return "No difference between A and B -> Practically Significant"

    # Otherwise, the result is inconclusive
    else:
        return "Inconclusive: needs more data"

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

def lost_based_decision(
    posterior_samples: np.ndarray,
    loss_weight_w: float = 2.5,
    null_value_s: float = 0.0,
    verbose: bool = True
):
    """
    Decide whether to launch based on posterior samples and loss-based decision rule.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Samples from the posterior distribution of the treatment effect.
    loss_weight_w : float
        Relative cost of being wrong when launching (w > 1 means launching is more risky).
    null_value_s : float
        The status quo effect size (typically 0).
    verbose : bool
        If True, prints summary of expected losses and decision.

    Returns
    -------
    decision : str
        'launch' or 'hold'
    decision_details : dict
        Dictionary with expected losses and posterior summary.
    """

    t_samples = posterior_samples
    t_mean = np.mean(t_samples)
    t_var = np.var(t_samples)

    # Expected loss if we do nothing
    expected_loss_hold = np.mean((t_samples - null_value_s) ** 2)

    # Expected loss if we launch based on posterior mean
    expected_loss_launch = loss_weight_w * np.mean((t_samples - t_mean) ** 2)

    decision = "launch" if expected_loss_launch < expected_loss_hold else "hold"

    details = {
        "decision": decision,
        "posterior_mean": t_mean,
        "posterior_std": np.std(t_samples),
        "expected_loss_hold": expected_loss_hold,
        "expected_loss_launch": expected_loss_launch,
        "loss_weight_w": loss_weight_w,
        "bias_squared": (t_mean - null_value_s)**2,
        "variance": t_var,
    }

    if verbose:
        print(f"Posterior mean = {t_mean:.4f}, std = {np.sqrt(t_var):.4f}")
        print(f"Expected loss (hold)   = {expected_loss_hold:.4f}")
        print(f"Expected loss (launch) = {expected_loss_launch:.4f} [w = {loss_weight_w}]")
        print(f"=> Decision: {decision.upper()}")

    return decision, details
