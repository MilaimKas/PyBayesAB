
#todo: add conjugate prior for known mean and known precision 

import numpy as np

from scipy.stats import norm, t
import scipy.interpolate as interpolate

import matplotlib.pyplot as plt
from matplotlib import animation

import src.helper as helper
import src.plot_functions as plot_functions

class Bays_norm:
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
        returns the posterior predictive distribution which gives the probabilty for the next observation
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

        return helper.norminvgamma(mu, kappa, alpha, beta) 

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
        post = helper.norminvgamma(m,k,a,b)
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
        post = helper.norminvgamma(m,k,a,b)
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
            post = helper.norminvgamma(m,k,a,b)
            post_pts = post.pdf(MU, S)

            anim_contourf = axs.contourf(MU,S, post_pts, cmap="Blues") 
            anim_contour = axs.contour(MU, S, post_pts, cmap="Blues")

            return anim_contourf, anim_contour


        # call the animator.  blit=True means only re-draw the parts that have changed.
        return animation.FuncAnimation(fig, animate,
                                frames=len(data), interval=interval, blit=False)