
#todo: add conjugate prior for known mean and known precision 

import numpy as np

from scipy.stats import norm, t, gamma

import matplotlib.pyplot as plt
from matplotlib import animation

from PyBayesAB.base_model import BayesianModel  
from PyBayesAB.base_plot import PlotManager  
from PyBayesAB import helper

from PyBayesAB import N_SAMPLE, N_PTS

class NormKnownMeanMixin:

    def __init__(self, mu, prior=None):

        self.mu = mu
        if prior is None:
            prior = [1, 50]

        self.prior = prior
        self.parameter_name = "Precision"

    def tau2sig(self, tau):
        return np.sqrt(1/tau) if tau > 0 else np.inf
    
    def sig2tau(self, sig):
        return 1/sig**2 if sig > 0 else np.inf
    
    def get_parameters(self, parameters, group, data):
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("Gamma posterior needs 2 parameters: alpha and beta")
            else:
                a,b = parameters
        else:
            a,b = self.post_parameters(group=group, data=data)
        return a, b

    def make_default_tau_range(self, a, b, percentile=0.999):
        """
        mean + variance as max
        """
        # Define the percentile bounds
        lower_percentile = 1-percentile
        upper_percentile = percentile

        # Calculate the meaningful range
        taumin = gamma.ppf(lower_percentile, a=a, scale=1/b)
        taumax = gamma.ppf(upper_percentile, a=a, scale=1/b)
        return [taumin, taumax]
    
    def add_rand_experiment(self, tau, n_data, group="A"):
        data_pts = norm.rvs(self.mu, scale=self.tau2sig(tau), size=n_data)
        self.add_experiment(data_pts, group=group)
    
    def post_pred(self, size=1, group="A"):
        a,b = self.post_parameters(group=group)
        df = 2 * a         
        loc = self.mu                
        scale = np.sqrt(b / a) 
        return t.rvs(df, loc=loc, scale=scale, size=size)
    
    def post_parameters(self, group="A", data=None):
        if data is None:
            data = self.return_data(group)
        a = self.prior[0]
        b = self.prior[1]
        for d in data:
            a += len(d)/2
            b += sum((d-self.mu)**2)/2
        return a, b

    def make_rvs(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE):
        a, b = self.get_parameters(parameters, group, data)
        return gamma.rvs(a, scale=1/b, size=N_sample)
    
    def make_pdf(self, parameters=None, data=None, group="A", p_pts=None, para_range=None):
        if group not in ["A", "B"]:
            raise ValueError("Group must be either 'A' or 'B' for pdf calculation.")
        a,b = self.get_parameters(parameters, group, data)
        if p_pts is None:
            if para_range is None:
                para_range = self.make_default_tau_range(a, b)
            p_pts = np.linspace(para_range[0], para_range[1], N_PTS)
        return gamma.pdf(p_pts, a, scale=1/b)

    def make_cum_post_para(self, group="A"):
        data = self.return_data(group)
        # cumulative alpha and beta value
        a_cum = []
        b_cum = []
        a=self.prior[0]
        b=self.prior[1]
        for i in range(len(data)):
            a += len(data[i])/2
            b += sum((data[i]-self.mu)**2)/2
            a_cum.append(a)
            b_cum.append(b)
        return a_cum, b_cum

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):
        # create list of rvs and pdf
        a_cum, b_cum = self.make_cum_post_para(group=group)
        rvs_data = []
        pdf_data = []
        if para_range is None:
            para_range = self.make_default_tau_range(a_cum[1], b_cum[1])
        p_pts = np.linspace(para_range[0], para_range[1], N_pts)
        for a,b in zip(a_cum, b_cum):
            rvs_data.append(self.make_rvs(parameters=[a,b], N_sample=N_sample))
            pdf_data.append(self.make_pdf(parameters=[a,b], p_pts=p_pts))
        return p_pts, rvs_data, pdf_data

class BaysNormKnownMean(NormKnownMeanMixin, BayesianModel, PlotManager):
    def __init__(self, mean, prior=None):
        BayesianModel.__init__(self)
        NormKnownMeanMixin.__init__(self,mu=mean, prior=prior)

class NormMixin:
    def __init__(self, prior=None):
        """
        Class for:
        - likelihood = normal distribution with unknown mean and precision
        - prior and posterior = GammaNorm

        Contains ploting and animation function

        Args:
            mu_prior (int, optional): _description_. Defaults to 0.
            kappa_prior (int, optional): _description_. Defaults to 1.
            alpha_prior (float, optional): _description_. Defaults to 0.5.
            beta_prior (int, optional): .Defaults to 50
        """

        # prior parameters
        if prior is None:
            mu_prior = 0
            kappa_prior = 1
            alpha_prior = 0.5
            beta_prior = 50
        else:
            if len(prior) != 4:
                raise ValueError("NormMixin prior needs 4 parameters: mu, kappa, alpha, beta")
            mu_prior, kappa_prior, alpha_prior, beta_prior = prior

        self.mu_prior = mu_prior
        self.kappa_prior = kappa_prior
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        self.data = []
        self.parameter_name = "Mean"
    
    def make_default_range(self, mu=None, alpha=None, beta=None, kappa=None, var="mu"):
        """
        returns the default range for mu, which is mean +/- 3*std
        and for std, which is mean +/- 3*std
        Args:
            mu (float, optional): mean. Defaults to None.
            alpha (float, optional): alpha parameter of the NormalGamma distribution. Defaults to None.
            beta (float, optional): beta parameter of the NormalGamma distribution. Defaults to None.
            kappa (float, optional): kappa parameter of the NormalGamma distribution. Defaults to None.
            var (str, optional): variable to return the range for. Defaults to "mu". Can be "mu", "std" or "both".
        """
        if mu is None:
            mu = self.mu_prior      
        if alpha is None or beta is None or kappa is None:
            alpha, beta, kappa = self.alpha_prior, self.beta_prior, self.kappa_prior
        mean = mu
        sig = np.sqrt(beta/(alpha*kappa))
        if var == "mu":
            return [mean - 3*sig, mean + 3*sig]
        elif var == "std":
            return [0, 3*sig]
        elif var == "both":
            return [[mean - 3*sig, mean + 3*sig], [0, 3*sig]]
        else:
            raise ValueError("var must be 'mu', 'std' or 'both'")
       
    def add_rand_experiment(self, n, mu, sig, group="A"):
        """
        add n random pts from a Normal distribution with mean mu and std sig
        to the data 

        Args:
            n (int): number of values
            mu (float): mean
            sig (float): standard deviation
        """
        experiment = norm.rvs(loc=mu, scale=sig, size=n)
        self.add_experiment(experiment, group=group)
    
    def post_pred(self, data=None):
        """
        returns the posterior predictive distribution which gives the probability for the next observation
        p(x|X) where x = new observation and X = all data collected so far.

        Posterior predictive distribution is the Student t distribution with dof 2*alpha
        
        Args:
            data (array): data values. If None, taken from self.data. Default is None.
        
        Returns:
            scipy.stats.t : posterior predictive distribution given the data
        """

        _,kappa,alpha,beta = self.post_parameters(data)
        if data is None:
            data = self.data
        mean = np.mean(data)

        loc = mean
        tau = (alpha*kappa)/(beta(kappa+1)) # precision
        sig = np.sqrt(1/tau) # std
        dof = 2*alpha
        return t(dof, loc=loc, scale=sig)
    
    def post_parameters(self, group="A", data=None):
        """
        return the parameter of the NormalGamma posterior given the (normal) data

        Returns:
            tuple: mu, kappa, alpha and beta value
        """

        if data is None:
            data = self.return_data(group)

        data_flat =  np.concatenate(data).ravel()
        n = len(data_flat)
        if n == 0:
            raise ValueError("No data available to calculate posterior parameters.")
            
        # Sample statistics
        sample_mean = np.mean(data_flat)
        sample_variance = np.var(data_flat, ddof=0)  # Use population variance
        
        # Posterior parameters
        kappa_post = self.kappa_prior + n
        mu_post = (self.kappa_prior * self.mu_prior + n * sample_mean) / kappa_post
        alpha_post = self.alpha_prior + n / 2
        beta_post = (self.beta_prior + 
                    0.5 * n * sample_variance + 
                    (self.kappa_prior * n * (sample_mean - self.mu_prior)**2) / (2 * kappa_post))
    
        return mu_post, kappa_post, alpha_post, beta_post

    def get_parameters(self, group,  parameters=None, data=None):
        """
        Get the parameters for the Normal Inverse Gamma distribution.
        If parameters are provided, they are used. Otherwise, the posterior parameters are calculated
        based on the group and data.
        """

        if parameters is not None:
            if len(parameters) != 4:
                raise ValueError(f"Normal Inverse Gamma posterior needs 4 parameters: mu, kappa, alpha, beta. Got {parameters}.")
            else:
                mu, kappa, alpha, beta = parameters
        else:
            mu, kappa, alpha, beta = self.post_parameters(group=group, data=data)
        
        return mu, kappa, alpha, beta
    
    def make_rvs(self, parameters=None, data=None, group="A", N_sample=N_SAMPLE, var="mu"):
        """
        Generate random variates from the Normal Inverse Gamma distribution.    
        """
        
        mu, kappa, alpha, beta = self.get_parameters(group, parameters, data)
        rvs = helper.NormInvGamma(mu, kappa, alpha, beta).rvs(size=N_sample)
        if var == "mu":
            return  rvs[0]
        elif var == "std":
            return  rvs[1]
        elif var == "both":
            return rvs    
         
    def make_pdf(self, parameters=None, data=None, group="A", p_pts=None, para_range=None, var="mu"):
        if group not in ["A", "B"]:
            raise ValueError("Group must be either 'A' or 'B' for pdf calculation.")
        mu, kappa, alpha, beta = self.get_parameters(group, parameters, data)
        p_pts = self._get_pts_range(p_pts, para_range, var)
        nig = helper.NormInvGamma(mu, kappa, alpha, beta)
        if var == "mu":
            return nig.marginal_mu_pdf(p_pts)
        elif var == "std":
            return nig.marginal_sigma_pdf(p_pts)
        elif var == "both":
            # joint pdf for mu and std
            return nig.pdf(p_pts[0], p_pts[1])
        
    def make_cum_post_para(self, group="A"):
        """
        Calculate the cumulative posterior parameters for the Normal Inverse Gamma distribution.
        """
        data = self.return_data(group)
        # cumulative mu, kappa, alpha, beta
        mu_cum =  [self.mu_prior]
        kappa_cum = [self.kappa_prior]
        a_cum = [self.alpha_prior]
        b_cum = [self.beta_prior]
        a=self.alpha_prior
        b=self.beta_prior
        mu = self.mu_prior
        kappa = self.kappa_prior
        for i in range(len(data)):
            n = len(data[i])
            mean = np.mean(data[i])
            mu = (kappa*mu + n*mean)/(kappa+n)
            kappa += n
            a += n/2
            b += 0.5*np.sum((data[i]-mean)**2) + (kappa*n*(mean-self.mu_prior)**2)/(2*kappa)
            mu_cum.append(mu)
            kappa_cum.append(kappa)
            a_cum.append(a)
            b_cum.append(b)
        return mu_cum, kappa_cum, a_cum, b_cum

    def make_cum_posterior(self, group="A", N_sample=N_SAMPLE, para_range=None, var="mu", N_pts=N_PTS):
        """
        Create cumulative posterior distributions for the Normal Inverse Gamma distribution.
        """
        # create list of rvs and pdf
        mu_cum, kappa_cum, a_cum, b_cum= self.make_cum_post_para(group=group)
        p_pts = self._get_pts_range(None, para_range, var)
        rvs_data = []
        pdf_data = []
        for mu, k, a,b in zip(mu_cum, kappa_cum, a_cum, b_cum):
            rvs_data.append(self.make_rvs(parameters=[mu,  k, a, b], N_sample=N_sample, var=var))
            pdf_data.append(self.make_pdf(parameters=[mu,  k, a, b], p_pts=p_pts, var=var))
        return p_pts, rvs_data, pdf_data
    
    def _get_pts_range(self, p_pts, para_range, var, N_pts=N_PTS):
        """
        Get the points range for the pdf based on the variable type.
        """
        if p_pts is None:
            if para_range is None:
                para_range = self.make_default_range(var=var)
                if var in ["mu", "std"]:
                    p_pts = np.linspace(para_range[0], para_range[1], N_pts)
                elif var == "both":
                    p_pts = np.meshgrid(np.linspace(para_range[0][0], para_range[0][1], N_pts),
                                    np.linspace(para_range[1][0], para_range[1][1], N_pts))
            else:
                if var in ["mu", "std"]:
                    if len(para_range) != 2:
                        raise ValueError("para_range must be a list of two values for mu and std")
                    p_pts = np.linspace(para_range[0], para_range[1], N_pts)
                elif var == "both":
                    if len(para_range) != 2 or len(para_range[0]) != 2 or len(para_range[1]) != 2:
                        raise ValueError("para_range must be a list of two lists, each containing the range for mu and std")
                    p_pts = np.meshgrid(np.linspace(para_range[0][0], para_range[0][1], N_pts),
                                    np.linspace(para_range[1][0], para_range[1][1], N_pts))
        else:
            if var in ["mu", "std"]:
                if p_pts.ndim != 1:
                    raise ValueError("p_pts must be a 1D array for mu or std")
            elif var == "both":
                if p_pts.ndim != 2 or p_pts[0].ndim != 1 or p_pts[1].ndim != 1:
                    raise ValueError("p_pts must be a 2D meshgrid for mu and std")

        return p_pts
    
class BaysNorm(NormMixin, BayesianModel, PlotManager):
    def __init__(self, prior=None):
        BayesianModel.__init__(self)
        NormMixin.__init__(self, prior=prior)



if __name__ == "__main__":

    import numpy as np

    mu_A = 20
    std_A = 10
    tau_A = 1/std_A**2 
    mu_B = 25
    std_B = 12
    tau_B = 1/std_B**2
    normal = BaysNorm()
    n_exp = 20
    for i in range(n_exp):
        n_data = np.random.randint(10,50)
        normal.add_rand_experiment(mu_A, std_A, n_data, group="A")
        normal.add_rand_experiment(mu_B, std_B, n_data, group="B")

    # calculate some Bayesian metrics

    ROPE = [-5, 5]  # Region of Practical Equivalence
    print()
    print(normal.summary_result(rope_interval=ROPE,level=95))
    print()
    
    # check shape of posterior
    print("Posterior parameters for group A:", normal.post_parameters(group="A"))
    print("Shape of mu posterior for group A:", normal.make_rvs(group="A").shape)
    print("HDI for mu group A:", normal.hdi(group="A", level=95))
    print("Shape of std posterior for group A:", normal.make_rvs(group="A", var="std").shape)
    print("HDI for std group A:", normal.hdi(group="A", level=95, post_kwargs={"var": "std"}))
    print()