import numpy as np

import scipy.optimize as optimize
from scipy import interpolate
from scipy.stats import gamma, norm, invgamma
from scipy.special import gamma as gamma_func, loggamma

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

import math

from PyBayesAB import STYLE


def hdi(distribution, level=0.95, norm_app=False):
	"""
	Get the highest density interval for the distribution, 
    e.g. for a Bayesian posterior, the highest posterior density interval (HPD/HDI)
    
    distribution = scipy.stats object
	"""

	if hasattr(distribution, 'ppf'):

		if norm_app:
			mean, std = distribution.stats()
			lower_limit, upper_limit = get_hdi_norm(mean, std, confidence_level=level)
		else:
			# For a given lower limit, we can compute the corresponding 95% interval
			def interval_width(lower):
				upper = distribution.ppf(distribution.cdf(lower) + level)
				return upper - lower
			
			# Find such interval which has the smallest width
			# Use equal-tailed interval as initial guess
			initial_guess = distribution.ppf((1-level)/2)
			optimize_result = optimize.minimize(interval_width, initial_guess)
			
			upper_limit = optimize_result.x[0]
			width = optimize_result.fun
			lower_limit = lower_limit + width
	
	elif isinstance(distribution, np.ndarray):
		
		if norm_app:
			mean, std = np.mean(distribution), np.std(distribution)
			lower_limit, upper_limit = get_hdi_norm(mean, std, confidence_level=level)
		else:
			n = len(distribution)

			interval_idx_inc = int(np.floor(level * n))
			n_intervals = n - interval_idx_inc
			interval_width = distribution[interval_idx_inc:] - distribution[:n_intervals]

			if len(interval_width) == 0:
				raise ValueError('Too few elements for interval calculation')

			min_idx = np.argmin(interval_width)
			upper_limit = distribution[min_idx]
			lower_limit = distribution[min_idx + interval_idx_inc]

	else:
		raise ValueError("Distribution must be either a function with the ppf method or a rvs (numpy array)")

	return (lower_limit, upper_limit)

def hdi_fromxy(x, y, hdi_prob=0.95):
	
    # Normalize y to be a probability density
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])  # Handle uneven spacing
    density = y / np.sum(y * dx)

    # Sort x and y by density
    sorted_indices = np.argsort(-y)  # Sort in descending order
    x_sorted = x[sorted_indices]
    density_sorted = y[sorted_indices]
    dx_sorted = dx[sorted_indices]

    # Compute cumulative probability
    cumulative_prob = np.cumsum(density_sorted * dx_sorted)

    # Find the HDI interval
    hdi_indices = []
    for i, prob in enumerate(cumulative_prob):
        if prob <= hdi_prob:
            hdi_indices.append(i)
    
    # Get HDI interval bounds
    x_min = np.min(x_sorted[hdi_indices])
    x_max = np.max(x_sorted[hdi_indices])

    return x_min, x_max

class KDE:
	def __init__(self, kde, x):
		"""
		Create a KDE class from a kde object by adding a cdf and ppf 
		"""
		self.x = x
		self.kde = kde
		self.cdf = np.vectorize(lambda p: kde.integrate_box_1d(-np.inf, p))
		cdf = self.cdf(x)
		self.ppf= interpolate.interp1d(cdf, x, kind='cubic', bounds_error=False)

		self.stats = self.get_mean_var()

	def get_mean_var(self):
			
		pdf = self.kde(self.x)

		# Normalize the PDF to ensure it integrates to 1
		dx = self.x[1] - self.x[0]
		pdf /= np.sum(pdf * dx)
		
		# Compute the mean
		mean = np.sum(self.x * pdf * dx)
		
		# Compute the variance
		variance = np.sum((self.x - mean)**2 * pdf * dx)
		std = np.sqrt(variance)
		
		return mean, std

class GammaCustom:
	def __init__(self, a=10**-3, b=1):
		"""
		wrapper for scipy.stats.gamma
		for alpha, beta parameters instead of a and scale
		"""
		self.alpha = a
		self.beta = b

	def pdf(self, model_para_pts, alpha=None, beta=None):
		if alpha is None:
			alpha = self.alpha
		if beta is None:
			beta = self.beta
		return gamma.pdf(model_para_pts, alpha, scale=1/beta)

	def rvs(self, size=None, alpha=None, beta=None):
		if alpha is None:
			alpha = self.alpha
		if beta is None:
			beta = self.beta
		return gamma.rvs(alpha, scale=1/beta, size=size)

	def ppf(self, model_para_pts, alpha=None, beta=None):
		if alpha is None:
			alpha = self.alpha
		if beta is None:
			beta = self.beta
		return gamma.ppf(model_para_pts, alpha, scale=1/beta)

	def cdf(self, model_para_pts, alpha=None, beta=None):
		if alpha is None:
			alpha = self.alpha
		if beta is None:
			beta = self.beta
		return gamma.cdf(model_para_pts, alpha, scale=1/beta)

class NormalGamma:
	def __init__(self, mu=0, kappa=1, a=0.5, b=50):
		"""

		class for the gamma-normal distribution 
		NG(x, lambda) = NormalGamma(mu, kappa, alpha, beta)
		ref: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf 

		Args:
		mu (_type_): location parameter
		kappa (float): sample size relative to the uncertainity in spread. Defaults to 1 (for non informative case)
		a (float): parameter of the Gamma component. Defaults to 1 (for non informative case)
		b (float): parameter of the Gamma component. Defaults to 1 (for non informative case).
		"""

		self.mu = mu
		self.kappa = kappa
		self.alpha = a
		self.beta = b

	def rvs(self, size=5000):
		"""
		random value sampler 
		lam is the precision (lam = 1/sig**2)

		Args:
			size (int, optional): Number of random variables to return. Defaults to 1000.
		Returns:
			[mu]*size, [sigma]*size
		"""
		lam = gamma.rvs(self.alpha, scale=1/self.beta, size=size)

		return np.asarray([norm.rvs(loc=self.mu, scale=np.sqrt(1/(l*self.kappa))) for l in lam]), np.asarray(np.sqrt(1/lam))

	def pdf(self, mu, sig):
		"""
		probability density function for the gamma-normal distribution

		Args:
			mu (float): loc parameter (mean)
			lambda (float): scale parameter (prescision)
		"""
		lam = 1/sig**2
		t1 = np.sqrt(self.kappa*lam/(2*np.pi))
		t2 = self.beta**(self.alpha)/gamma_func(self.alpha)
		t3 = lam**(self.alpha+1)
		t4 = np.exp(-lam/2*(2*self.beta+lam*(mu-self.mu)**2))

		return t1*t2*t3*t4

class NormInvGamma():
    """A normal inverse gamma random variable.
    
    The Normal-Inverse-Gamma distribution is a conjugate prior for 
    normal distributions with unknown mean and variance.
    
    Parameters:
    -----------
    mu : float
        Location parameter (prior mean for the normal mean)
    kappa : float  
        Precision parameter (how much we trust the prior mean)
    alpha : float
        Shape parameter for inverse gamma (degrees of freedom / 2)
    beta : float
        Scale parameter for inverse gamma
        
    Notes
    -----
    The probability density function for `norminvgamma` is:
    .. math::
        f(μ, σ² | μ₀, κ, α, β) = 
            √(κ/(2πσ²)) * (β^α/Γ(α)) * (1/σ²)^(α+1) * 
            exp(-(2β + κ(μ - μ₀)²)/(2σ²))
    """
    
    def __init__(self, mu, kappa, alpha, beta):
        self.mu = mu        # μ₀ - prior mean
        self.kappa = kappa  # κ - precision parameter  
        self.alpha = alpha  # α - shape parameter
        self.beta = beta    # β - scale parameter
        
        # Validate parameters
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")  
        if beta <= 0:
            raise ValueError("beta must be positive")
    
    def rvs(self, size=1000):
        """
        Random value sampler for the Normal Inverse Gamma distribution.
        
        Args:
            size (int, optional): number of samples. Defaults to 1000.
            
        Returns:
            tuple: (mu_samples, sigma_samples) where:
                - mu_samples: array of sampled means
                - sigma_samples: array of sampled standard deviations
        """
        # Step 1: Sample σ² from Inverse-Gamma(α, β)
        sigma_squared = invgamma.rvs(a=self.alpha, scale=self.beta, size=size)
        
        # Step 2: Sample μ from Normal(μ₀, σ²/κ) for each σ²
        mu_samples = np.array([
            norm.rvs(loc=self.mu, scale=np.sqrt(s2/self.kappa)) 
            for s2 in sigma_squared
        ])
        
        # Return both mu and sigma (not sigma²)
        sigma_samples = np.sqrt(sigma_squared)
        
        return mu_samples, sigma_samples
    
    def pdf(self, mu, sigma):
        """
        Probability density function for the Normal Inverse Gamma distribution.
        
        Args:
            mu (float or array): The mean parameter values
            sigma (float or array): The standard deviation parameter values
            
        Returns:
            float or array: The probability density function values
        """
        # Use logpdf and exponentiate for numerical stability
        return np.exp(self.logpdf(mu, sigma))
    
    def logpdf(self, mu, sigma):
        """
        Log probability density function (numerically stable).
        
        Args:
            mu (float or array): The mean parameter values  
            sigma (float or array): The standard deviation parameter values
            
        Returns:
            float or array: The log probability density function values
        """
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        sigma_squared = sigma**2
        
        # Handle edge cases
        if np.any(sigma <= 0):
            return np.full_like(sigma, -np.inf)
        
        # Calculate log PDF components using stable computations
        log_t1 = 0.5 * (np.log(self.kappa) - np.log(2 * np.pi) - np.log(sigma_squared))
        log_t2 = self.alpha * np.log(self.beta) - loggamma(self.alpha)  # Use loggamma!
        log_t3 = -(self.alpha + 1) * np.log(sigma_squared)
        log_t4 = -(2*self.beta + self.kappa*(mu - self.mu)**2) / (2*sigma_squared)
        
        return log_t1 + log_t2 + log_t3 + log_t4
    
    def pdf_stable(self, mu, sigma):
        """
        Numerically stable PDF computation with overflow protection.
        
        Args:
            mu (float or array): The mean parameter values
            sigma (float or array): The standard deviation parameter values
            
        Returns:
            float or array: The probability density function values
        """
        logpdf_val = self.logpdf(mu, sigma)
        
        # Clip extremely negative values to prevent underflow
        logpdf_val = np.clip(logpdf_val, -700, 700)  # exp(-700) ≈ 10^-304
        
    def marginal_mu_pdf(self, mu_values):
        """
        Marginal PDF for mu (integrated over sigma).
        This follows a t-distribution: t_{2α}(μ₀, β/(α·κ))
        
        Args:
            mu_values (float or array): Values at which to evaluate the marginal PDF
            
        Returns:
            float or array: Marginal PDF values for mu
        """
        from scipy.stats import t
        
        mu_values = np.asarray(mu_values)
        
        # Parameters for the t-distribution
        df = 2 * self.alpha  # degrees of freedom
        loc = self.mu        # location (mean)
        scale = np.sqrt(self.beta / (self.alpha * self.kappa))  # scale
        
        return t.pdf(mu_values, df=df, loc=loc, scale=scale)
    
    def marginal_sigma_pdf(self, sigma_values):
        """
        Marginal PDF for sigma.
        This is derived from the inverse-gamma distribution for sigma²
        
        Args:
            sigma_values (float or array): Values at which to evaluate the marginal PDF
            
        Returns:
            float or array: Marginal PDF values for sigma
        """
        sigma_values = np.asarray(sigma_values)
        
        # Handle edge cases
        if np.any(sigma_values <= 0):
            result = np.zeros_like(sigma_values, dtype=float)
            result[sigma_values <= 0] = 0.0
            valid_mask = sigma_values > 0
            if np.any(valid_mask):
                sigma_valid = sigma_values[valid_mask]
                # PDF of sigma from sigma² ~ InvGamma(α, β)
                # If Y ~ InvGamma(α, β), then √Y has PDF: 2y * InvGamma_pdf(y², α, β)
                sigma_squared = sigma_valid**2
                invgamma_pdf = invgamma.pdf(sigma_squared, a=self.alpha, scale=self.beta)
                result[valid_mask] = 2 * sigma_valid * invgamma_pdf
            return result
        else:
            sigma_squared = sigma_values**2
            invgamma_pdf = invgamma.pdf(sigma_squared, a=self.alpha, scale=self.beta)
            return 2 * sigma_values * invgamma_pdf
    
    def marginal_sigma_logpdf(self, sigma_values):
        """
        Log marginal PDF for sigma (numerically stable).
        """
        sigma_values = np.asarray(sigma_values)
        
        if np.any(sigma_values <= 0):
            return np.full_like(sigma_values, -np.inf)
        
        sigma_squared = sigma_values**2
        invgamma_logpdf = invgamma.logpdf(sigma_squared, a=self.alpha, scale=self.beta)
        return np.log(2) + np.log(sigma_values) + invgamma_logpdf
	
def make_range(rvs):
	min = np.min(rvs)
	max = np.max(rvs)
	wid = (max - min)/2
	para_range = [min-wid, max+wid]
	return para_range

def get_parameters_pts(rvs, n_pts):
	min = np.min(rvs)
	max = np.max(rvs)
	return np.linspace(min, max, n_pts)

def get_ticks(ticks, max_ticks=10):
	if len(ticks) > max_ticks:
		step = math.ceil(len(ticks)/max_ticks)
		return np.arange(0, len(ticks), step), ticks[::step]
	else:
		return np.arange(len(ticks)), ticks 

def truncate_colormap(name, minval=0, maxval=1.0, n=100):
	cmap = plt.get_cmap(name)
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def flatten_nested_list(nested_list):
	flat_list = []
	for arr in nested_list:
		flat_list += list(arr)
	return flat_list

def get_hdi_norm(mean, std, confidence_level):
	alpha = 1 - confidence_level
	z = norm.ppf(1 - alpha / 2)
	lower_bound = mean - z * std
	upper_bound = mean + z * std
	return lower_bound, upper_bound

class MplColorHelper:
	def __init__(self, cmap_name, start_val, stop_val):
		plt.style.use(STYLE)
		self.cmap_name = cmap_name
		self.cmap = plt.get_cmap(cmap_name)
		self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
		self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

	def get_rgb(self, val):
		return tuple(map(float, self.scalarMap.to_rgba(val)))

def darken_color(color, factor=0.9):
    """
    Darkens an RGBA color by scaling its RGB values.
    
    Args:
        color (tuple): An RGBA color tuple, where each value is in [0, 1].
        factor (float): A factor to darken the color, should be in [0, 1].
    
    Returns:
        tuple: A darkened RGBA color.
    """
    return (color[0] * factor, color[1] * factor, color[2] * factor, color[3])

def create_colormap_from_rgba(color, darker_factor=1, range=[0, 0.7]):
    """
    Creates a colormap object that transitions from the given RGBA color to white.
    
    Args:
        color (tuple): An RGBA color tuple, e.g., (r, g, b, a) where each value is in [0, 1].
    
    Returns:
        LinearSegmentedColormap: A colormap transitioning from the given color to white.
    """
    color_darker = darken_color(color, factor=darker_factor)
    # Define the color map using the input color and white
    cdict = {
        'red':   [(0.0, 1.0, 1.0), (1.0, color_darker[0], color_darker[0])],
        'green': [(0.0, 1.0, 1.0), (1.0, color_darker[1], color_darker[1])],
        'blue':  [(0.0, 1.0, 1.0), (1.0, color_darker[2], color_darker[2])],
        'alpha': [(0.0, 1.0, 1.0), (1.0, color_darker[3], color_darker[3])],
    }
    full_cmap = LinearSegmentedColormap('custom_full_colormap', cdict)
    
	# Slice the colormap
    sliced_cmap = LinearSegmentedColormap.from_list(
        'custom_sliced_colormap',
        full_cmap(np.linspace(range[0], range[1], 256))
    )
    return sliced_cmap