
import numpy as np

from PyBayesAB import plot_functions
from PyBayesAB import N_SAMPLE, N_PTS, FIGSIZE


class PlotManager:
    
    #def __init__(self):

    def plot_final_posterior(self, group, N_sample=N_SAMPLE, N_pts=N_PTS, para_range=None, **kwargs):
        """
        plot the posterior distribution for the total result

        Args:
            n_rvs (int, optional): number of random value for the histogram. Defaults to 1000.
            prange (list, optional): [lower, upper] limit for p. Defaults to None.
        """       

        # check kwargs if any
        
        try: 
            parameter_name = self.parameter_name
        except:
            parameter_name = "Parameters"

        # plot either group A or group B
        if (group == "A") or (group == "B"):
            rvs = self.make_rvs(group=group, N_sample=N_sample)
            para_range = para_range or self.get_parameter_range(rvs)
            x_pdf = np.linspace(para_range[0], para_range[1], N_pts)
            y_pdf = self.make_pdf(group=group, p_pts=x_pdf)
            fig = plot_functions.plot_posterior([rvs], [x_pdf, [y_pdf]], labels=[group], xlabel=parameter_name, 
                                                **kwargs)

        # plot difference of posteriors
        elif group == "diff":
            rvs_A = self.make_rvs(group="A", N_sample=N_sample) 
            rvs_B = self.make_rvs(group="B", N_sample=N_sample)
            rvs_diff = rvs_A-rvs_B
            fig = plot_functions.plot_posterior([rvs_diff], labels=["A-B"], xlabel="Difference in "+parameter_name, 
                                                **kwargs)

        # plot both posterior of group A and group B
        elif group == "AB":
            rvs_A = self.make_rvs(group="A", N_sample=N_sample)
            rvs_B = self.make_rvs(group="B", N_sample=N_sample)
            para_range = para_range or self.get_parameter_range(np.concatenate((rvs_A, rvs_B)))
            x_pdf = np.linspace(para_range[0], para_range[1], N_pts)
            pdf_A = self.make_pdf(group="A", p_pts=x_pdf)
            pdf_B = self.make_pdf(group="B", p_pts=x_pdf)
            fig = plot_functions.plot_posterior([rvs_A, rvs_B], [x_pdf, [pdf_A, pdf_B]], labels=["A", "B"], xlabel=parameter_name, 
                                                **kwargs)

        else:
            raise SyntaxError("group can only be A, B, diff or AB")

        return fig
    
    def plot_cum_posterior(self, group, type, N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS):

        try: 
            parameter_name = self.parameter_name
        except:
            parameter_name = "Parameters"

        # get list of appropriate posteriors
        rvs_data, pdf_data = self.get_post_data(group=group, para_range=para_range, N_sample=N_sample, N_pts=N_pts)

        # plot posterior for one group
        if (group == "A") or (group == "B") or (group == "AB"):
            
            # 2D map plot
            if type == "2D":
                fig = plot_functions.plot_cumulative_posterior_2D_pdf(pdf_data, ylabel=parameter_name)
            
            # 1D plot
            elif type == "1D":
                fig = plot_functions.plot_cumulative_posterior_1D(rvs_data, pdf_data=pdf_data, xlabel=parameter_name)
        
        # plot posterior for the difference
        elif group == "diff":

            if type == "2D":
                fig = plot_functions.plot_cumulative_posterior_2D_rvs(rvs_data, ylabel=parameter_name)
                
            elif type == "1D":
                fig = plot_functions.plot_cumulative_posterior_1D(rvs_data, pdf_data=None, xlabel=parameter_name)
            
        else:
            raise ValueError("group must be either 'A', 'B', 'diff' or 'AB'")

        return fig

    def plot_anim(self, group, N_sample=N_SAMPLE, para_range=None, N_pts=N_PTS, interval=200, figsize=FIGSIZE):
        rvs_data, pdf_data = self.get_post_data(group=group, para_range=para_range, N_sample=N_sample, N_pts=N_pts)
        if pdf_data is None:
            pdf_data = rvs_data
        return plot_functions.animate_posterior(pdf_data, interval=interval, figsize=figsize)


    def get_post_data(self, group, para_range=None, N_sample=N_SAMPLE, N_pts=N_PTS):

        if (group == "A") or (group == "B"):
            
            param_pts, rvs_data, pdf_data = self.make_cum_posterior(group=group, N_sample=N_sample, para_range=para_range, N_pts=N_pts)
            rvs_data = [[r] for r in rvs_data]
            pdf_data = [param_pts, [[p] for p in pdf_data]]

        elif group == "AB":
            _ , rvs_data_A, pdf_data_A = self.make_cum_posterior(group="A", N_sample=N_sample, para_range=para_range, N_pts=N_pts)
            param_pts, rvs_data_B, pdf_data_B = self.make_cum_posterior(group="B", N_sample=N_sample, para_range=para_range, N_pts=N_pts)

            rvs_data = list(zip(rvs_data_A, rvs_data_B))
            pdf_data = [param_pts, list(zip(pdf_data_A, pdf_data_B))]

        elif group == "diff":

            _, rvs_data_A, _ = self.make_cum_posterior(group="A")
            _, rvs_data_B, _ = self.make_cum_posterior(group="B")

            rvs_data = []
            for rvs_a,rvs_b in zip(rvs_data_A, rvs_data_B):
                rvs_data.append([rvs_a-rvs_b])
            pdf_data = None
        
        return rvs_data, pdf_data
        

    def get_parameter_range(self, rvs):
        return [np.min(rvs), max(rvs)]