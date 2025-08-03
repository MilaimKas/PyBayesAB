import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

import matplotlib as mpl
import seaborn as sns

import numpy as np

from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

from PyBayesAB.config import N_BINS, N_PTS, FIGSIZE, STYLE, COLOR_MAP, LINEWIDTH
from PyBayesAB import helper

# define global style
plt.style.use(STYLE)

# make rgba list of colors from cmap
COL = helper.MplColorHelper(COLOR_MAP, 0, 1)
COLORS = [COL.get_rgb(i) for i in np.linspace(0,1,4)]

# list of cmaps
CMAPS = [helper.create_colormap_from_rgba(c) for c in COLORS] 

def plot_data(data, label, xlabel="Experiments", ylabel="Values",
              figsize=FIGSIZE, plot_kwargs={"linewidth":LINEWIDTH, "marker":"o", "markersize":5}):
    """ Plot data from experiments as a line plot.  
    Args:
        data (list of np.array): List of arrays, each containing values for one experiment.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple): Size of the figure.
        colors (list): List of colors for each group.
        labels (list of str): Labels for each group.
        plot_kwargs (dict): Additional keyword arguments for the plot.
    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = [f"Group {label} {i + 1}" for i in range(len(data))]
    for d, lab in zip(data, labels):
        ax.plot(d, label=lab, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    #plt.close()
    return fig

def plot_posterior(rvs, pdf=None, xlabel="Parameter", labels=None, 
                   figsize=FIGSIZE, colors=COLORS, bins=N_BINS, 
                   sns_hist_kwargs={"alpha":0.6, "element":"step", "edgecolor":None}, 
                   plot_kwargs={"linewidth":LINEWIDTH},
                   para_range=None):
    """
    Plot posterior distributions as histograms and/or PDFs.
    Args:
        rvs (list of np.array): Random values from posterior distributions.
        pdf (list of tuple): List of (x, y) for PDFs, or None. Must be the same length as rvs.
        xlabel (str): Label for the x-axis.
        labels (list of str): Labels for each group.
    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    labels = labels or [f"Group {i + 1}" for i in range(len(rvs))]

    if pdf:
        x, post_pdf = pdf
        if len(post_pdf) != len(rvs):
            raise ValueError(f"Number of posterior distribution must be the same for rvs and pdf, got {len(rvs)} and {len(post_pdf)}, respectively")
        for i, (samples, y, color) in enumerate(zip(rvs, post_pdf, colors)):
            label = labels[i]
            ax.plot(x, y, color=color, **plot_kwargs)
            sns.histplot(samples, bins=bins, stat="density", color=color, label=label, kde=False, ax=ax, **sns_hist_kwargs)
    else:
        for i, (samples, color) in enumerate(zip(rvs, colors)):
            label = labels[i]
            sns.histplot(samples, bins=bins, stat="density", color=color, label=label, ax=ax, **sns_hist_kwargs)
            sns.kdeplot(samples, color=color, **plot_kwargs)

    if para_range is not None:
        ax.set_xlim(para_range)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    #plt.close()

    return fig

def plot_cumulative_posterior_1D(rvs_data, pdf_data=None, plt_cm=CMAPS,  
                    xlabel="Parameter", bins=N_BINS, figsize=FIGSIZE, labels=None,
                    sns_hist_kwargs={"alpha":0.5, "element":"step", "kde_kws":{'linewidth': LINEWIDTH}, "edgecolor":None},
                    plot_kwargs={"linewidth":LINEWIDTH}, 
                    group_labels=None,
                    para_range=None):
    
    fig, ax = plt.subplots(figsize=figsize)
    N_exp = len(rvs_data)
    
    if labels is None:
        labels = np.arange(1, N_exp+1)

    cmaps = [plt_cm[0](np.linspace(0.1, 1, N_exp)), 
             plt_cm[1](np.linspace(0.1, 1, N_exp))]

    if pdf_data:
        x, post_pdf = pdf_data
        for i in range(N_exp):
            for j, (post_rvs, lab, pdf, xx) in enumerate(zip(rvs_data[i], group_labels, post_pdf[i], x)):
                col = cmaps[j][i]
                # plot rvs with large transparency 
                sns.histplot(post_rvs, bins=bins, color=col, ax=ax, stat="density", **sns_hist_kwargs)
                # plot pdf
                ax.plot(xx, pdf, color=col, **plot_kwargs, label=lab)
    
    else:
        for i in range(N_exp):
            for cmp, post_rvs in zip(cmaps, rvs_data[i]):
                # plot rvs with large transparency and kde 
                sns.histplot(post_rvs, bins=bins, color=cmp[i], ax=ax, stat="density", **sns_hist_kwargs)
                sns.kdeplot(post_rvs,  color=cmp[i], **plot_kwargs)

    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=plt_cm[0]),
             ax=ax, orientation='vertical', label="Experiments")
    
    if para_range is not None:
        ax.set_xlim(para_range)

    #plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Probability density")
    #plt.close()
    
    return fig

def plot_cumulative_posterior_2D_pdf(
    pdf_data, 
    figsize=FIGSIZE, 
    cmaps=CMAPS, 
    colors=COLORS, 
    ylabel="Parameter", 
    group_labels=["A", "B"],
    contour_kwargs={"levels":3, "alpha":1},
    colormesh_kwargs={"alpha":0.7}, 
    clabel_kwargs={"inline":True, "fontsize":8},
    para_range=None):

    fig, ax = plt.subplots(figsize=figsize)
    plt.tight_layout()

    param_pts, post_pdf = pdf_data

    ngroups = len(post_pdf[0])
    nx = len(post_pdf)
    x=np.arange(1, nx+1)

    pcolormesh_list = []
    # loop over groups
    for i in range(ngroups):
        ny = len(param_pts[i])
        X,Y =  np.meshgrid(x, param_pts[i])
        data_to_plot = np.zeros((nx*ny,3))
        # loop over experiment
        for j in range(nx):
            data_to_plot[j*ny:(j+1)*ny,0] = np.ones(ny)*x[j] # x
            data_to_plot[j*ny:(j+1)*ny,1] = param_pts[i] # y
            data_to_plot[j*ny:(j+1)*ny,2] = post_pdf[j][i] # z

        # create meshgrid
        Z = griddata((data_to_plot[:,0], data_to_plot[:,1]), data_to_plot[:,2], (X,Y), method='nearest')
        cp = ax.contour(X,Y,Z, colors=[colors[i]]*contour_kwargs["levels"], **contour_kwargs)
        ax.clabel(cp, **clabel_kwargs)
        c_map = cmaps[i]
        pcm = plt.pcolormesh(X,Y,Z, cmap=c_map, shading='auto', zorder=2, **colormesh_kwargs)
        pcolormesh_list.append(pcm)  # Store pcolormesh for colorbars

    # Add individual colorbars for each pcolormesh
    for i, pcm in enumerate(pcolormesh_list):
        cbar = fig.colorbar(pcm, ax=ax, label=f"Group {group_labels[i]} Probability Density", orientation="vertical", pad=0.01)
        cbar.ax.tick_params(labelsize=8)

    # find optimal y range given the posterior pdf
    if para_range is None:
        para_range = helper.get_optimal_xrange(post_pdf[1], param_pts)
    ax.set_ylim(para_range)

    plt.xticks(ticks=x, labels=[int(xx) for xx in x])
    plt.xlabel("Experiments")
    plt.ylabel(ylabel)
    #plt.close()
    #plt.legend()

    return fig

def plot_cumulative_posterior_3D(rvs_data, pdf_data=None,  
                    xlabel="Parameter", bins=30, figsize=(10, 7), labels=None,
                    plt_cm=CMAPS,
                    plt_bar_kwargs={"alpha":0.5},
                    plot_kwargs={"linewidth":1}, 
                    group_labels=None, 
                    para_range=None,
                    scaled_space=4, view=(40, -50)):
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')   
    ax.grid(False)
     
    N_exp = len(rvs_data)
    N_groups = len(rvs_data[0])

    if labels is None:
        labels = [str(i) for i in np.arange(1, N_exp+1)]
    
    if group_labels is None:
        group_labels = [f"group_{i}" for i in range(N_groups)]

    # Generate colormap for different experiments
    if isinstance(plt_cm, list) and len(plt_cm) >= 2:
        cmaps = [plt_cm[0](np.linspace(0.1, 1, N_exp)), 
                 plt_cm[1](np.linspace(0.1, 1, N_exp))]
    else:
        cmaps = [plt_cm(np.linspace(0.1, 1, N_exp))]  # Single colormap

    if pdf_data:
        x, post_pdf = pdf_data
        for i in range(N_exp):
            for j, (post_rvs, lab, pdf, xx) in enumerate(zip(rvs_data[i], group_labels, post_pdf[i], x)):
                col = cmaps[j][i]
                
                # Compute histogram
                hist, edges = np.histogram(post_rvs, bins=bins, density=True)
                bar_pos = [(a+edges[i+1])/2.0 for i,a in enumerate(edges[0:-1])]
                w = abs(bar_pos[1]) - abs(bar_pos[0])
                
                # Bar plot
                ax.bar(bar_pos, hist, zs=i, zdir='x', align='center', width=w, color=col, **plt_bar_kwargs)
                
                # PDF plot
                ax.plot(xx, pdf, zs=i, zdir='x', color=cmaps[j][-1], **plot_kwargs, label=lab)

    else:
        for i in range(N_exp):
            for j, (post_rvs, lab) in enumerate(zip(rvs_data[i], group_labels)):
                col = cmaps[j][i]

                # Compute histogram
                hist, edges = np.histogram(post_rvs, bins=bins, density=True)
                bar_pos = [(a+edges[i+1])/2.0 for i,a in enumerate(edges[0:-1])]
                w = abs(bar_pos[1]) - abs(bar_pos[0])

                # Corrected: Bar positioning
                ax.bar(bar_pos, hist, zs=i, zdir='x', align='center', width=w, color=col, **plt_bar_kwargs)

                # add gaussian kde plot
                pdf = gaussian_kde(post_rvs).pdf(bar_pos)
                ax.plot(bar_pos, pdf, zs=i, zdir='x', color=cmaps[j][-1], **plot_kwargs, label=lab)

    # add exp labels
    ax.set_xticks(np.arange(N_exp))
    ax.set_xticklabels(labels)

    ax.set_xlabel("Experiments")
    ax.set_ylabel(xlabel)

    # set y range 
    if para_range is not None:
        ax.set_ylim(para_range)

    # hide z axis and xy plane
    ax.zaxis.set_label_position('none')
    ax.zaxis.set_ticks_position('none')    
    ax.xaxis.pane.set_visible(False)  # Hides the Z-axis background
    ax.yaxis.pane.set_visible(False)

    #ax.legend()

    # stretch x axis
    ax.view_init(*view)  # Prevents cut-off issues
    ax.set_box_aspect(aspect=(scaled_space, scaled_space/2, 1))

    #plt.close()
    return fig


def plot_cumulative_posterior_2D_rvs(
    rvs_data, 
    exp_label=None, 
    bins=20, 
    ylabel="Parameter", 
    group_labels=["A", "B"],
    figsize=FIGSIZE,
    cmaps=CMAPS, 
    colors=COLORS, 
    contour_kwargs={"levels": 3, "alpha": 1},
    colormesh_kwargs={"alpha": 0.7}, 
    clabel_kwargs={"inline": True, "fontsize": 8},
    para_range=None
):
    """
    Plot a 2D colormesh using samples of posteriors for successive experiments.

    Args:
        rvs_data (list of np.array): List of arrays, each containing random samples for one "experiment."
        exp_label (list or np.array, optional): List of experiment indices or labels (x-axis).
        bins (int or array-like): Number of bins or specific bin edges for the histograms.
        ylabel (str, optional): Label for the y-axis (parameters). Defaults to "Parameter".
        param_range (list, optional): Range for the y-axis. If None, it will be determined from the data.
    """

    fig, ax = plt.subplots(figsize=figsize)

    nx = len(rvs_data)
    ngroups = len(rvs_data[0])

    # Set up x-axis labels
    if exp_label is None:
        exp_label = np.arange(1, nx + 1)

    # Compute histograms for each experiment
    bin_edges = None
    bin_centers = None
    Z_list = []  # List to store Z data for each group

    for i in range(ngroups):
        histograms = []
        for j in range(nx):
            hist, edges = np.histogram(rvs_data[j][i], bins=bins, density=True)
            histograms.append(hist)
            if bin_edges is None:
                bin_edges = edges  # Use the bin edges from the first histogram
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Midpoints of bins

        # Stack histograms into a 2D array for plotting
        Z = np.array(histograms).T  # Shape: (len(bin_centers), nx)
        Z_list.append(Z)

        # Prepare the grid for experiments (x-axis) and parameter bins (y-axis)
        X, Y = np.meshgrid(exp_label, bin_centers)

        # Colormesh plot
        pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmaps[i], **colormesh_kwargs)

        # Add colorbar
        cbar = fig.colorbar(pcm, ax=ax, label=f"Group {group_labels[i]} Probability Density", orientation="vertical", pad=0.01)

        # Add contour plot
        cp = ax.contour(X, Y, Z, colors=[colors[i]]*contour_kwargs["levels"], zorder=2, **contour_kwargs)
        ax.clabel(cp, **clabel_kwargs)

    # make  sure y is optimal
    if para_range is None:
        para_range = [bin_centers.min(), bin_centers.max()]
    ax.set_ylim(para_range)

    # Add labels and title
    ax.set_xticks(exp_label)
    ax.set_xlabel("Experiments")
    ax.set_ylabel(ylabel)
    #plt.close()

    return fig

def plot_bayesian_metrics(num_experiments, hdi_lower, hdi_upper, rope_values, map_values, prob_best, 
                        xlabel="Number of Experiments", ylabel="Metrics Value",
                        title="Bayesian Metrics", rope_interval=None):
    """
    Plots Bayesian metrics (HDI, ROPE, MAP) against the number of experiments using Matplotlib. 
    """

    ## HDI and MAP ##

    # Create Matplotlib figure
    fig1, ax = plt.subplots(figsize=(10, 6))

    # Plot HDI
    ax.plot(num_experiments, hdi_lower, label='_nolegend_', linestyle='--', color='blue')
    ax.plot(num_experiments, hdi_upper, label='_nolegend_', linestyle='--', color='blue')
    ax.fill_between(num_experiments, hdi_lower, hdi_upper, color='blue', alpha=0.2, label='HDI Range')

    # Plot MAP
    ax.plot(num_experiments, map_values, label='MAP', linestyle='-', color='red')

    if rope_interval is not None:
        # Plot ROPE as a horizontal line
        ax.fill_between(num_experiments, rope_interval[0], rope_interval[1],
                        color='green', alpha=0.2, label='ROPE Range')
    
    # add line  at  0
    ax.axhline(0, color='black', linestyle='--', linewidth=1, label="_nolegend_")   

    # make sure y range is optimal. Not accounting for prior
    ymin = min(hdi_lower[1:])
    ymax = max(hdi_upper[1:])
    ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

    # Add labels and legend
    ax.set_title('Bayesian Metrics vs Number of Experiments')
    ax.set_xlabel('Number of Experiments')
    ax.set_ylabel('Metric Value')
    ax.legend()

    ### ROPE and Probability of Best ###
    fig2, ax2 = plt.subplots(figsize=(10, 6))   

    # Plot ROPE probability
    ax2.plot(num_experiments, rope_values, label='ROPE', linestyle='-', color='green')
    
    # Plot probability of best
    ax2.plot(num_experiments, prob_best, label='Probability of A better than B', linestyle='-', color='orange')

    # Add labels and legend
    ax2.set_title('Bayesian Metrics vs Number of Experiments')
    ax2.set_xlabel('Number of Experiments')
    ax2.set_ylabel('Metric Value')
    ax2.legend()

    return fig1, fig2

def animate_posterior(post_data, interval=200, 
                        figsize=FIGSIZE, colors=COLORS,
                        group_labels=["A", "B"],
                        kwargs_post={"linewidth":2, "edgecolor":(0,0,0,1)}, 
                        kwargs_hdis= {"linewidth":2, "edgecolor":(0,0,0,0.7), "alpha":0.7},
                        xlabel="Parameters", labels=None, n_pts=N_PTS, exp_label=None,
                        norm_app=True,
                        para_range=None):
    
    plt.rcParams["animation.html"] = "jshtml"

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.tight_layout(pad=2)

    if len(post_data) == 2:
        type = "pdf"
        param_pts, post_pts = post_data
        n_exp = len(post_pts)
        ymax = max(helper.flatten_nested_list(post_pts[-1])) 
        ymin = min(helper.flatten_nested_list(post_pts[0]))
        if para_range is None:
            para_range = [min(min(p) for p in param_pts), max(max(p) for p in param_pts)]
    else:
        type = "rvs"
        post_pts = post_data.copy()
        n_exp = len(post_pts)
        ymax = max(np.histogram(helper.flatten_nested_list(post_pts[-1]), bins=100, density=True)[0])
        ymin = min(np.histogram(helper.flatten_nested_list(post_pts[0]), bins=100, density=True)[0])
        if para_range is None:
            para_range = [min(helper.flatten_nested_list(post_pts[0])), max(helper.flatten_nested_list(post_pts[0]))]
        param_pts = [np.linspace(para_range[0], para_range[1], n_pts)]
      
    # posterior distribution(s) and hdis
    lines = []
    hdis =[]
    for c, post, p_pts in zip(colors, post_pts[0], param_pts):
        if type == "pdf":
            l = axs[0].fill_between(p_pts, post, color=c, **kwargs_post)
        else:
            l = axs[0].fill_between(p_pts, gaussian_kde(post)(p_pts), color=c, **kwargs_post)
        hdis.append(axs[1].fill_between(np.arange(n_exp), np.zeros(n_exp), color=c, **kwargs_hdis))
        lines.append(l)

    axs[0].set_ylabel("Density")
    axs[0].set_xlabel(xlabel)
    axs[1].set_ylabel(xlabel)
    axs[1].set_xlabel("Experiments")

    axs[0].set_xlim(para_range[0], para_range[1])
    axs[1].set_xlim(0, n_exp)

    axs[0].set_ylim(ymin, ymax)
    axs[1].set_ylim(para_range[0], para_range[1])

    if labels is None:
        labels = [f"Group {group_labels[i]}" for i in range(len(lines)) ]

    axs[0].legend(lines, labels=labels)
    axs[1].legend(lines, labels=labels)

    if exp_label is None:
        exp_label = np.arange(1, n_exp + 1)

    # dont show plot on notebook
    plt.close(fig) 
        
    def update(frame):
        for i in range(len(lines)):
            
            # update post
            # Get the paths created by fill_between
            path_post = lines[i].get_paths()[0]
            verts_post = path_post.vertices
            path_hdis = hdis[i].get_paths()[0]
            verts_hdis = path_hdis.vertices

            # Elements 1:Nx+1 encode the upper curve
            if type == "pdf":
                post = post_pts[frame][i]
                up, low = helper.hdi_fromxy(param_pts[i], post)
                verts_post[1:len(param_pts[i])+1, 1] = post
            else:
                up, low = helper.hdi(post_pts[frame][i], norm_app=norm_app)
                verts_post[1:len(param_pts[i])+1, 1] = gaussian_kde(post_pts[frame][i])(param_pts[i])

            verts_hdis[frame+1, 1] = up
            verts_hdis[-frame-1, 1] = low

        axs[1].set_xlim(0, max(1, frame-2))
        x_ticks_pos, x_ticks_labels = helper.get_ticks(exp_label[:max(1, frame)])
        axs[1].set_xticks(x_ticks_pos, labels=x_ticks_labels)

    return animation.FuncAnimation(fig, update, frames=n_exp, interval=interval, blit=False)

