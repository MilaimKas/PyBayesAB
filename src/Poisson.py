
import numpy as np

from scipy.stats import gamma, poisson, nbinom

import src.helper as helper
import src.plot_functions as plot_functions

N_SAMPLE=5000
N_PTS = 1000
N_BINS = 20

class Bays_poisson:
    def __init__(self, prior_a=1, prior_b=1) -> None:
        """
        Class for:
        - likelihood = Poisson
        - prior and posterior = Gamma

        Contains ploting and animation function
        """

        self.data = []
        self.dataB = []

        self.prior_a = prior_a
        self.prior_b = prior_b

    def add_experiment(self, n_events, group="A"):
        """
        add 1 interval with nbr of event

        Args:
            n_events (int): number of occurences or events for one interval
        """
        if group == "A":
            self.data.append([n_events])
        elif group == "B":
            self.data.append([n_events])
    
    def add_rand_experiment(self,n,mu, group="A"):
        """
        add n intervals with random nbr of events ([n_events])
        taken from a Poisson distribution with meam mu

        Args:
            n (in): number of intervals
            mu (float): mean number of occurences
        """
        n_events = poisson.rvs(mu, size=n)
        if group == "A":
            self.data.extend(n_events)
        elif group == "B":
            self.dataB.extend(n_events)
    
    def post_pred(self, group="A"):
        """
        returns the posterior predictve distribution which gives the probabilty for the next observation
        p(x|X) where x = new observation and X = all data collected so far

        Returns:
            scipy.stats.nbinom: posterior predictive distribution
        """
        a,b = self.post_parameters(group=group)
        return nbinom(a,b/(1+b))
    
    def post_parameters(self, group="A", data=None):
        """
        return the parameters for the gamma posterior given the data
        """
        if data is None:
            if group == "A":
                data = self.data
            elif group == "B":
                data = self.dataB
        a = sum(data)+self.prior_a
        b = len(data)+self.prior_b

        return a,b

    def make_rvs(self, data=None, group="A", N_sample=N_SAMPLE):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            group (str, optional): _description_. Defaults to "A".
            N_sample (_type_, optional): _description_. Defaults to N_SAMPLE.

        Returns:
            _type_: _description_
        """
        a,b = self.post_parameters(group=group, data=data)
        return gamma.rvs(a, scale=1/b, size=N_sample)
    
    def make_pdf(self, data=None, group="A", mu_range=[0,50], N_pts=N_PTS):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            group (str, optional): _description_. Defaults to "A".
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.
            N_pts (_type_, optional): _description_. Defaults to N_PTS.

        Returns:
            _type_: _description_
        """
        a,b = self.post_parameters(group=group, data=data)
        mu_pts = np.linspace(mu_range[0], mu_range[1], N_pts)
        return mu_pts, gamma.pdf(mu_pts, a, scale=1/b)

    def make_cum_post_para(self, group="A"):
        """_summary_

        Args:
            group (str, optional): _description_. Defaults to "A".

        Returns:
            _type_: _description_
        """
        
        if group == "A":
            data = self.data
        elif group == "B":
            data = self.dataB

        data.insert(0, self.prior_a)
        data = np.array(data, dtype="object")

        # cumulative events
        a_cum = np.cumsum(data)
        b_cum = np.zeros(len(data)+1)
        b_cum[0] = self.prior_b
        b_cum[1:] = np.arange(1, len(data)+1)

        return a_cum, b_cum
    
    def plot_tot(self, n_rvs=N_SAMPLE, mu_range=None, group="A", n_pts=N_PTS, data=None):
        """
        plot the posterior distribution for the total result

        Args:
            n_rvs (int, optional): number of random values for the histogram. Defaults to 1000.
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.
        """
        
        if data is None:
            
            if (group == "A") or (group == "B"):
                rvs = self.make_rvs(group=group, N_sample=n_rvs)
                if mu_range is None:
                    mu_range = [np.min(rvs), np.max(rvs)]
                model_para_pts, post = self.make_pdf(group=group, mu_range=mu_range)
                fig = plot_functions.plot_tot([rvs], model_para_pts, [post], 
                                              labels=[group], xlabel="Mean number of occurences")
            
            elif group == "diff":
                
                rvs_A = self.make_rvs(group="A", N_sample=n_rvs) 
                rvs_B = self.make_rvs(group="B", N_sample=n_rvs)
                rvs_diff = rvs_A-rvs_B
                if mu_range is None:
                    mu_range = [np.min(rvs_diff), np.max(rvs_diff)]
                model_para_pts = np.linspace(mu_range[0], mu_range[1], n_pts)
                fig = plot_functions.plot_tot([rvs_diff],model_para_pts, 
                                              labels=["A-B"], xlabel="Difference mean number of occurences")
            
            elif group == "AB":
                rvs_A = self.make_rvs(group="A", N_sample=n_rvs)
                rvs_B = self.make_rvs(group="B", N_sample=n_rvs)
                if mu_range is None:
                    rvs_tmp = np.concatenate((rvs_A, rvs_B))
                    mu_range = [np.min(rvs_tmp), np.max(rvs_tmp)]
                model_para_pts, post_A = self.make_pdf(group="A", mu_range=mu_range)
                _, post_B = self.make_pdf(group="B", mu_range=mu_range)
                fig = plot_functions.plot_tot([rvs_A, rvs_B], model_para_pts, [post_A, post_B],
                                              labels=["A", "B"], 
                                              xlabel="Mean number of occurences")    
            else:
                raise SyntaxError("group can only be A,B,diff or AB")
        
        else:
            raise NotImplementedError

        return fig

    def plot_exp(self, type="1D", n_pdf=N_PTS, n_rvs=N_SAMPLE, mu_range=None, group="A"):
        """
        plot "cumulative" posteriors

        Args:
            type (str, optional): one dimensional plot with x=p and y=pdf 
                                    or 2 dimensional plot with x=exp and y=p_range and z=pdf. 
                                    Defaults to "1D".
            n_pdf (int, optional): Number of points on the x axis. Defaults to 1000.
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.

            """
            
        gamma_post = helper.gamma_custom

        n_exp = len(self.data)+1
        exp = np.arange(1, n_exp+1)

        if (group == "A") or (group == "B"):
            if mu_range is None:
                rvs = self.make_rvs(group=group)
                mu_range = helper.make_range(rvs)
            mu_pts = np.linspace(mu_range[0], mu_range[1], n_pdf)
            zip_post_para = [zip(*self.make_cum_post_para(group=group))]
            labels = [group]
            if type == "2D":
                fig = plot_functions.plot_cum_post_2D_pdf(gamma_post, zip_post_para, labels, 
                                        exp, mu_pts, 
                                        post_para_label="Mean number of occurences")
            elif type == "1D":
                fig = plot_functions.plot_cum_post_1D_pdf(gamma_post, zip_post_para, labels, 
                                                          exp, mu_pts, 
                                                          post_para_label="Mean number of occurences")
                
        elif group == "AB":
            zip_post_para = [zip(*self.make_cum_post_para(group="A")), 
                                zip(*self.make_cum_post_para(group="B"))]
            if mu_range is None:
                rvs = np.concatenate((self.make_rvs(), self.make_rvs(group="B")))
                mu_range = helper.make_range(rvs)
            mu_pts = np.linspace(mu_range[0], mu_range[1], n_pdf)
            labels = ["A", "B"]
            if type == "2D":
                fig = plot_functions.plot_cum_post_2D_pdf(gamma_post, zip_post_para, labels, 
                                        exp, mu_pts, 
                                        post_para_label="Mean number of occurences")
            elif type == "1D":
                fig = plot_functions.plot_cum_post_1D_pdf(gamma_post, zip_post_para, labels, 
                                        exp, mu_pts, 
                                        post_para_label="Mean number of occurences")
                
        elif group == "diff":
            A_a, A_b = self.make_cum_post_para(group="A")
            B_a, B_b = self.make_cum_post_para(group="B")
            rvs_diff = self.make_rvs()-self.make_rvs(group="B")
            range = helper.make_range(rvs_diff)
            if mu_range is None:
                mu_range = range
            mu_pts = np.linspace(mu_range[0], mu_range[1], n_pdf)
            hist_list = []
            rvs_list = []
            for aa,ab,ba,bb in zip(A_a, A_b, B_a, B_b):
                rvs = gamma_post(alpha=aa, beta=ab).rvs(n_rvs)-gamma_post(alpha=ba, beta=bb).rvs(n_rvs)
                hist = np.histogram(rvs, bins=N_BINS, range=range, density=True)[0]
                hist_list.append(hist)
                rvs_list.append(rvs)

            if type == "2D":
                hist_arr = np.array(hist_list)
                fig = plot_functions.plot_cum_post_2D_rvs(hist_arr, range,
                                                          ylabel="mu(A)-mu(B)")
            elif type == "1D":
                fig = plot_functions.plot_cum_post_1D_rvs([rvs_list], exp, 
                                                          model_para_pts=mu_pts, labels=["A-B"],
                                                          post_para_label="Difference of mean number of occurences")
        else:
            raise NotImplementedError
        
        return fig

    def plot_anim(self, mu_range=None, n_pdf=N_PTS, n_rvs=N_SAMPLE, interval=None, 
                  list_hdi=[95,90,80,60], group="A"):
        """
        Create an animation for the evolution of the posterior

        Args:
            mu_range (list, optional): [lower, upper] limit for mu. Defaults to None.
            n_pdf (int, optional): number of pts for the Poisson means. Defaults to 1000.
            interval (float, optional): time in ms between frames. Defaults to None.
            list_hdi (list, optional): list of hdi's to be displayed. Defaults to [95,90,80,60].

        Returns:
            pyplot.animate.Funcanimation
        """

        gamma_post = helper.gamma_custom
    
        if (group == "A") or (group == "B"):
            post_para = [(a,b) for a,b in zip(*self.make_cum_post_para(group=group))]
            if mu_range is None:
                rvs = self.make_rvs(group=group)
                mu_range = helper.make_range(rvs)
            return plot_functions.plot_anim_pdf(gamma_post, post_para, mu_range, 
                model_para_label="Mean number of occurences", 
                list_hdi=list_hdi, n_pdf=n_pdf, interval=interval)

        elif group == "diff":
            A_a, A_b = self.make_cum_post_para(group="A")
            B_a, B_b = self.make_cum_post_para(group="B")
            rvs_list = []
            rvs_diff = self.make_rvs()-self.make_rvs(group="B")
            if mu_range is None:
                mu_range = helper.make_range(rvs_diff)
            for aa,ab,ba,bb in zip(A_a, A_b, B_a, B_b):
                rvs = gamma(aa, scale=1/ab).rvs(size=N_SAMPLE)-gamma(ba, scale=1/bb).rvs(size=N_SAMPLE)
                rvs_list.append(rvs)
            
            return plot_functions.plot_anim_rvs(rvs_list, mu_range=mu_range)