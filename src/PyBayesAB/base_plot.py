import numpy as np

from PyBayesAB import plot_functions, helper, bayesian_functions
from PyBayesAB.config import N_SAMPLE, N_PTS, FIGSIZE


class PlotManager:
    
    #def __init__(self):

    def plot_final_posterior(self, group="diff", N_sample=N_SAMPLE, N_pts=N_PTS, para_range=None, plot_kwargs={}, post_kwargs={}):
        """
        plot the posterior distribution for the total result

        Args:
            n_rvs (int, optional): number of random value for the histogram. Defaults to 1000.
            prange (list, optional): [lower, upper] limit for p. Defaults to None.
        """       

        parameter_name = self.get_parameter_name()

        # plot either group A or group B
        if (group == "A") or (group == "B"):
            rvs = self.make_rvs(group=group, N_sample=N_sample, **post_kwargs)
            para_range = para_range or self.get_parameter_range(rvs)
            x_pdf = np.linspace(para_range[0], para_range[1], N_pts)
            # make pdf for the given group if make_pdf is implemented
            if hasattr(self, 'make_pdf'):
                y_pdf = self.make_pdf(group=group, p_pts=x_pdf, **post_kwargs)
                dist_list = [rvs, [x_pdf, y_pdf]]
            else:
                dist_list = [rvs]
            fig = plot_functions.plot_posterior(dist_list, labels=[group], xlabel=parameter_name, 
                                                **plot_kwargs)

        # plot difference of posteriors
        elif group == "diff":
            rvs_diff = self.make_rvs_diff(N_sample=N_sample, post_kwargs=post_kwargs)
            fig = plot_functions.plot_posterior([rvs_diff], labels=["A-B"], xlabel="Difference in "+parameter_name, 
                                                **plot_kwargs)

        # plot both posterior of group A and group B
        elif group == "AB":
            rvs_A = self.make_rvs(group="A", N_sample=N_sample,**post_kwargs)
            rvs_B = self.make_rvs(group="B", N_sample=N_sample,  **post_kwargs)
            para_range = para_range or self.get_parameter_range(np.concatenate((rvs_A, rvs_B)))
            x_pdf = np.linspace(para_range[0], para_range[1], N_pts)
            # make pdf for both groups if make_pdf is implemented
            if not hasattr(self, 'make_pdf'):
                pdf_A = self.make_pdf(group="A", p_pts=x_pdf, **post_kwargs)
                pdf_B = self.make_pdf(group="B", p_pts=x_pdf, **post_kwargs)
                dist_list = [rvs_A, rvs_B], [x_pdf, [pdf_A, pdf_B]]
            else:
                dist_list = [rvs_A, rvs_B]
            fig = plot_functions.plot_posterior(dist_list, labels=["A", "B"], xlabel=parameter_name, 
                                                **plot_kwargs)

        else:
            raise SyntaxError("group can only be A, B, diff or AB")

        return fig
    
    def plot_cum_posterior(self, group="diff", type="1D", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS, **post_kwargs):

        parameter_name = self.get_parameter_name()

        # get list of appropriate posteriors
        rvs_data, pdf_data = self.get_post_data(group=group, para_range=para_range, N_sample=N_sample, N_pts=N_pts, **post_kwargs)

        # plot posterior for one group
        if (group == "A") or (group == "B") or (group == "AB"):
            
            if group == "AB":
                group_labels = ["A","B"]
            else:
                group_labels = group

            # 2D map plot
            if type == "2D":
                fig = plot_functions.plot_cumulative_posterior_2D_pdf(pdf_data, ylabel=parameter_name, group_labels=group_labels, para_range=para_range)
            
            # 1D plot
            elif type == "1D":
                fig = plot_functions.plot_cumulative_posterior_1D(rvs_data, pdf_data=pdf_data, xlabel=parameter_name, group_labels=group_labels, para_range=para_range)

            elif type == "3D":
                fig = plot_functions.plot_cumulative_posterior_3D(rvs_data, pdf_data=pdf_data, xlabel=parameter_name, group_labels=group_labels, para_range=para_range)
        
        # plot posterior for the difference
        elif group == "diff":

            if type == "2D":
                fig = plot_functions.plot_cumulative_posterior_2D_rvs(rvs_data, ylabel=parameter_name, group_labels=["diff"], para_range=para_range)
                
            elif type == "1D":
                fig = plot_functions.plot_cumulative_posterior_1D(rvs_data, pdf_data=None, xlabel=parameter_name, group_labels=["diff"], para_range=para_range)

            elif type == "3D":
                fig = plot_functions.plot_cumulative_posterior_3D(rvs_data, xlabel=parameter_name, group_labels=["diff"], para_range=para_range)
            
        else:
            raise ValueError("group must be either 'A', 'B', 'diff' or 'AB'")

        return fig

    def plot_anim(self, group="diff", N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS, interval=200, figsize=FIGSIZE, **post_kwargs):

        parameter_name = self.get_parameter_name()

        rvs_data, pdf_data = self.get_post_data(group=group, para_range=para_range, N_sample=N_sample, N_pts=N_pts, **post_kwargs)
        if pdf_data is None:
            pdf_data = rvs_data
        return plot_functions.animate_posterior(pdf_data, interval=interval, figsize=figsize, xlabel=parameter_name, para_range=para_range)


    def get_post_data(self, group="diff", para_range=None, N_sample=N_SAMPLE, N_pts=N_PTS, **post_kwargs):
        """
        post process the posterior data for the given group to be used in plotting

        Args:
            group (str): _group_ can be "A", "B", "AB" or "diff"
            para_range (list, optional): range of the model's parameters. Defaults to None.
            N_sample (int, optional): number of samples to draw  from rvs. Defaults to N_SAMPLE.
            N_pts (array, optional): number of parameter pts for pdf. Defaults to N_PTS.

        Returns:
            rvs list, pdf list
        """

        # check missing data
        self._check_missing_data()

        if (group == "A") or (group == "B"):
            
            param_pts, rvs_data, pdf_data = self.make_cum_posterior(group=group, N_sample=N_sample, para_range=para_range, N_pts=N_pts, **post_kwargs)
            rvs_data = [[r] for r in rvs_data]
            pdf_data = [[param_pts], [[p] for p in pdf_data]]

        elif group == "AB":
            param_pts_A , rvs_data_A, pdf_data_A = self.make_cum_posterior(group="A", N_sample=N_sample, para_range=para_range, N_pts=N_pts, **post_kwargs)
            param_pts_B, rvs_data_B, pdf_data_B = self.make_cum_posterior(group="B", N_sample=N_sample, para_range=para_range, N_pts=N_pts, **post_kwargs)

            rvs_data = list(zip(rvs_data_A, rvs_data_B))
            pdf_data = [[param_pts_A, param_pts_B], list(zip(pdf_data_A, pdf_data_B))]

        elif group == "diff":

            rvs_data = self.make_cum_rvs_diff(N_sample=N_sample, **post_kwargs)
            pdf_data = None
        
        return rvs_data, pdf_data

    def get_parameter_range(self, rvs):
        return [np.min(rvs), max(rvs)]
    
    def get_parameter_name(self):
        """
        checkif base class has parameter_name attribute
        if not, use default name "Parameters"
        """
        if hasattr(self, 'parameter_name'):
            parameter_name = self.parameter_name
        else:
            parameter_name = "Parameters"
        return parameter_name

    def plot_bayesian_metrics(self, rope_interval, N_sample=N_SAMPLE, **post_kwargs):
        """
        Plots Bayesian metrics (HDI, ROPE, MAP) against the number of experiments using Matplotlib.

        Args:
            group (str, optional): The group to plot metrics for ("A", "B" or "diff"). Defaults to "A".
            N_sample (int, optional): Number of samples to generate. Defaults to N_SAMPLE.

        Returns:
            matplotlib.figure.Figure: The Matplotlib figure with Bayesian metrics.
        """

        # Initialize lists to store metrics
        hdi_lower = []
        hdi_upper = []
        rope_values = []
        map_values = []
        prob_best = []
        num_experiments = list(range(0, len(self.return_data("A")) + 1))

        # get posteriors
        rvs_data, _ = self.get_post_data(group="diff", N_sample=N_sample, para_range=None, **post_kwargs)
        
        # Calculate metrics for each number of experiments
        for rvs in rvs_data:
            
            # HDI
            hdi_low, hdi_up = helper.hdi(rvs, level=0.95)
            hdi_lower.append(hdi_low)
            hdi_upper.append(hdi_up)

            # ROPE
            rope_values.append(bayesian_functions.rope(rvs=rvs, interval=rope_interval)*100)

            # MAP
            map_values.append(bayesian_functions.map(rvs,  method='kde'))

            # prob best
            prob_best.append(bayesian_functions.prob_best(rvs))

        fig1,  fig2 =  plot_functions.plot_bayesian_metrics(
            num_experiments, hdi_lower, hdi_upper, rope_values, map_values, prob_best,
            xlabel="Number of Experiments", ylabel="Metrics Value", 
            title=f"Bayesian Metrics for A Vs B", rope_interval=rope_interval
            )

        return fig1,  fig2
    
    def plot_data(self, group="A", **plot_kwargs):
        """
        Plot the data for a given group.

        Args:
            group (str, optional): The group to plot data for ("A" or "B"). Defaults to "A".

        Returns:
            matplotlib.figure.Figure: The Matplotlib figure with the data plot.
        """
        if group not in ["A", "B"]:
            raise ValueError("Group must be either 'A' or 'B'.")

        data = self.return_data(group)

        if "label" not in plot_kwargs:
            plot_kwargs["label"] = group
            
        fig = plot_functions.plot_data(data, **plot_kwargs)

        return fig