
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
        self.parameter_name = "Prescision"

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

class BaysNormKnownSTDMixin:
    def __init__(self) -> None:
        raise NotImplementedError()

    def add_experiment(self, pts):
        return
    
    def add_rand_experiment(self,n, mean):
        return
    
    def post_pred(self, data=None):
        return
    
    def post_parameters(self,data=None):
        return
    
    def post_distr(self,data=None):
        return
    
    def plot_tot(self, mean_lower, mean_upper, data=None, n_pdf=1000):
        return
    
    def plot_anim(self, mean_lower, mean_upper, n_pdf=1000, data=None, interval=None):
        return
    

class BaysNormal:
    def __init__(self, mu_prior=0, kappa_prior=1, alpha_prior=0.5, beta_prior=50):
        """
        Class for:
        - likelihood = normal distribution with unknown mean and prescision
        - prior and posterior = GammaNorm

        Contains ploting and animation function

        Args:
            mu_prior (int, optional): _description_. Defaults to 0.
            kappa_prior (int, optional): _description_. Defaults to 1.
            alpha_prior (float, optional): _description_. Defaults to 0.5.
            beta_prior (int, optional): .Defaults to 50
        """

        self.mu_prior = mu_prior
        self.kappa_prior = kappa_prior
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        self.data = []
    
    def add_experiment(self, pts):
        """
        add an experiment (array of "measured" values) to the data

        Args:
            pts (np.array): array with measured values
        """
        self.data.append(pts)
    
    def add_rand_experiment(self,n,mu,sig):
        """
        add n random pts from a Normal distribution with mean mu and std sig
        to the data 

        Args:
            n (int): number of values
            mu (float): mean
            tau (float): precision
        """
        self.data.append(norm.rvs(loc=mu, scale=sig, size=n))

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
    
    def post_parameters(self,data=None):
        """
        return the parameter of the NormalGamma posterior given the data

        Returns:
            tuple: mu, kappa, alpha and beta value
        """

        if data is None:
            n = len(self.data)
            data = np.array( self.data, dtype="object").flatten()[0]
        else:
            n = len(data)

        mean = np.mean(data)

        kappa = self.kappa_prior+n
        mu = (self.kappa_prior*self.mu_prior + n*mean)/kappa
        alpha = self.alpha_prior+n/2
        beta = self.beta_prior + 0.5*np.sum((data-mean)**2) + (self.kappa_prior*n*(mean-self.mu_prior)**2)/(2*kappa)

        return mu, kappa, alpha, beta
    
    def post_distr(self,data=None):
        """
        Return a Normal-Gamma distribution object with method pdf and rvs
        using the parameters calculated from data.

        Args:
            data (array, optional): list of data values. If None, take self.data. Defaults to None.

        Returns:
            helper.Normgamma object
        """
        
        mu, kappa, alpha, beta = self.post_parameters(data)

        return helper.NormInvGamma(mu, kappa, alpha, beta) 

    def plot_tot(self, mu_lower, mu_upper, sig_lower, sig_upper, data=None, n_pdf=1000):
        """
        Return a countour plot with the pdf as a function of mu and sig

        Args:
            mu_lower (float): lower value of mean to be plotted.
            mu_upper (float): upper value of mean to be plotted.
            sig_lower (float): lower value of standard deviation to be plotted.
            sig_upper (float): upper value of standard deviation to be plotted.
            data (list of np.array, optional): list of "experiments" containing the collected data. 
                                               If None, self.data will be used. Defaults to None.
            n_pdf (int, optional): number of pts for the pdf plot. Defaults to 1000.
        """

        mu = np.linspace(mu_lower,mu_upper,n_pdf)
        sig = np.linspace(sig_lower,sig_upper,n_pdf)

        MU, S = np.meshgrid(mu,sig)

        m,k,a,b = self.post_parameters(data=data)
        #post = helper.normalgamma(m,k,a,b)
        post = helper.NormInvGamma(m,k,a,b)
        post_pts = post.pdf(MU, S)

        plt.contourf(MU, S, post_pts, cmap="Blues")
        plt.pcolormesh(MU, S, post_pts, shading='auto', cmap="Blues", alpha=0.7)
        plt.colorbar(label="Probabilty density")
        plt.show()
    
    def plot_anim(self, mu_lower, mu_upper, sig_lower, sig_upper, n_pdf=1000, data=None, interval=None):
        
        if data is None:
            data = self.data

        plt.rcParams["animation.html"] = "jshtml"

        if interval is None:
            # default is 5s duration
            interval = 5000/len(data)

        # frame
        fig, axs = plt.subplots(1)
        fig.tight_layout()

        m,k,a,b = self.post_parameters(data=data[0])
        post = helper.NormInvGamma(m,k,a,b)
        mu = np.linspace(mu_lower,mu_upper,n_pdf)
        sig = np.linspace(sig_lower,sig_upper,n_pdf)
        MU, S = np.meshgrid(mu,sig)
        post_pts = post.pdf(MU, S)

        # initialize posterior plot  
        axs.set_xlabel("Mean")
        axs.set_ylabel("Standard deviation")
        axs.set_xlim(mu_lower,mu_upper)
        axs.set_ylim(sig_lower, sig_upper)   
        #axs.colorbar(label="Probabilty density")
  
        anim_contourf = axs.contourf(MU, S, post_pts, cmap="Blues")
        anim_contour = axs.contour(MU, S, post_pts, cmap="Blues")

        plt.close(fig)  # dont show initial plot

        # animation function  
        def animate(i):
            # ref: https://brushingupscience.com/2019/08/01/elaborate-matplotlib-animations/
            nonlocal anim_contourf, anim_contour

            for cp,cm in zip(anim_contourf.collections, anim_contour.collections):
                cp.remove()
                cm.remove()
            
            # update posetrior
            m,k,a,b = self.post_parameters(data=np.concatenate(data[0:i+1]).ravel())
            post = helper.NormInvGamma(m,k,a,b)
            post_pts = post.pdf(MU, S)

            anim_contourf = axs.contourf(MU,S, post_pts, cmap="Blues") 
            anim_contour = axs.contour(MU, S, post_pts, cmap="Blues")

            return anim_contourf, anim_contour


        # call the animator.  blit=True means only re-draw the parts that have changed.
        return animation.FuncAnimation(fig, animate,
                                frames=len(data), interval=interval, blit=False)