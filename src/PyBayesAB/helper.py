import numpy as np

import scipy.optimize as optimize
from scipy import interpolate
from scipy.stats import gamma, norm, gengamma, expon
from scipy.special import gamma as gamma_func


def hdi(distribution, level=0.95):
	"""
	Get the highest density interval for the distribution, 
    e.g. for a Bayesian posterior, the highest posterior density interval (HPD/HDI)
    
    distribution = scipy.stats object
	"""

	if callable(distribution.ppf):
		# For a given lower limit, we can compute the corresponding 95% interval
		def interval_width(lower):
			upper = distribution.ppf(distribution.cdf(lower) + level)
			return upper - lower
		
		# Find such interval which has the smallest width
		# Use equal-tailed interval as initial guess
		initial_guess = distribution.ppf((1-level)/2)
		optimize_result = optimize.minimize(interval_width, initial_guess)
		
		lower_limit = optimize_result.x[0]
		width = optimize_result.fun
		upper_limit = lower_limit + width
	
	elif isinstance(distribution, np.ndarray):
		
		n = len(distribution)

		interval_idx_inc = int(np.floor(level * n))
		n_intervals = n - interval_idx_inc
		interval_width = distribution[interval_idx_inc:] - distribution[:n_intervals]

		if len(interval_width) == 0:
			raise ValueError('Too few elements for interval calculation')

		min_idx = np.argmin(interval_width)
		lower_limit = distribution[min_idx]
		upper_limit = distribution[min_idx + interval_idx_inc]

	else:
		raise ValueError("Distribution must be either a function with the ppf method or a numpy array")

	return (lower_limit, upper_limit)


class KDE:
	def __init__(self, kde, xi, xf, xn):
		x = np.linspace(xi, xf, xn)
		self.kde = kde
		self.cdf = np.vectorize(lambda p: kde.integrate_box_1d(-np.inf, p))
		cdf = self.cdf(x)
		self.ppf= interpolate.interp1d(cdf, x, kind='cubic', bounds_error=False)


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
	The mu (``mu``) keyword specifies the parmaeter mu.
	Notes
	-----
	The probability density function for `norminvgamma` is:
	.. math::
		x = [\mu, \sigma^2]
		f(x | \delta, \alpha, \beta, \lamda) = 
				\sqrt(\frac{\lamda}{2 \pi x[\sigma^2}])
				\frac{\beta^\alpha}{\gamma(\alpha)}
				\frac{1}{x[\sigma^2]}^(\alpha + 1)
				\exp(- \frac{2\beta + \lamda(x[\mu] - delta)^2}{2 x[\sigma^2] })
		
	for a real number :math:`x` and for positive number :math: `\sigma^2` > 0

	ref: https://deebuls.github.io/devblog/probability/python/plotting/matplotlib/2020/05/19/probability-normalinversegamma.html
	"""

	def __init__(self, mu, kappa, a, b):

		self.mu = mu
		self.alpha = a
		self.beta = b
		self.kappa = kappa

	def rvs(self, size=1000):
		sigma_2 = gengamma.rvs(self.alpha, self.beta,  size=size)
		sigma_2 = np.array(sigma_2)
		return [norm.rvs(self.mu, scale=np.sqrt(s/self.kappa)) for s in sigma_2], np.sqrt(sigma_2)

	def pdf(self, mu, sig):
		t1 = ((self.kappa)**0.5) * ((self.beta)**self.alpha)
		t2 = (sig * (2 * 3.15)**0.5) * gamma_func(self.alpha)
		t3 = (1 / sig**2)**(self.alpha + 1)
		t4 = expon.pdf((2*self.beta + self.kappa*(self.mu-mu)**2)/(2*sig**2))
		return (t1/t2)*t3*t4
	
def make_range(rvs):
	min = np.min(rvs)
	max = np.max(rvs)
	wid = (max - min)/2
	para_range = [min-wid, max+wid]
	return para_range