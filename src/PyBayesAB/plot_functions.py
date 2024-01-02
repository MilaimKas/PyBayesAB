
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as colors

import numpy as np
import scipy.interpolate as interpolate
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import scipy.ndimage

from PyBayesAB import helper

FIGSIZE=(6,4)
COLORS = ["red", "blue", "green", "orange"]
N_BINS = 20
N_SAMPLE = 5000
N_PTS = 2000


def plot_tot(rvs, model_para_pts, pdf=None, xlabel="Model parameter", labels=[None]):
    """
    Plot the probability density and histogram for a posterior distribution

    Args:
        rvs (np.array): array with the random values for the histogramm.
        pdf (dict): "model_para_pts" - array with axis 0 the parameter value and axis 1 the probability density.
        xlabel (strin, optional): label for the x axis (model parameter)
    """
    
    if pdf is None:
        pdf = [None]*len(rvs)

    fig = plt.figure(figsize=FIGSIZE)

    for r,p,lab,c in zip(rvs, pdf, labels, COLORS):
        #plot rvs
        plt.hist(r, density=True, alpha=0.4, label=lab, color=c, bins=N_BINS)
        # plot pdf
        if p is not None:
            plt.plot(model_para_pts, p, color=c)
        # plot gaussian kde
        else:
            kde = gaussian_kde(r)
            plt.plot(model_para_pts, kde(model_para_pts), color=c)

    plt.xlabel(xlabel)
    plt.ylabel("Probability density")
    plt.legend()
    
    return fig


def plot_cum_post_2D_pdf(post, zip_post_para, labels, exp, model_para, post_para_label="Parameter"):
    """
    Plot a colormesh 2D plot with x=succesive posterior ('experiments')

    Args:
        post (scipy.stats): object conatining the posterior distribution function. Must have a pdf method.
        zip_post_para (zip object): _description_
        exp (_type_): _description_
        model_para (_type_): _description_
        n_pdf (_type_): _description_
        ylabel (str, optional): _description_. Defaults to "Parameter".
    """

    fig = plt.figure(figsize=FIGSIZE)

    n_exp = len(exp)
    n_pdf = len(model_para)
    data_pts = np.zeros((n_pdf*n_exp,3))
    cmaps = ["Reds", "Blues"]
    color = ["red", "blue"] 
    X,Y =  np.meshgrid(exp, model_para)

    if len(labels) > 1:
        alpha = 0.7
    else:
        alpha = 1

    def truncate_colormap(name, minval=0, maxval=1.0, n=100):
        cmap = plt.get_cmap(name)
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    c = 0
    # loop over groups
    for z, lab in zip(zip_post_para, labels):
        for i,post_para in enumerate(z):
            Post = post(*post_para)
            post_pdf = Post.pdf(model_para)
            data_pts[i*n_pdf:(i+1)*n_pdf,0] = np.ones(n_pdf)*exp[i]
            data_pts[i*n_pdf:(i+1)*n_pdf,1] = model_para
            data_pts[i*n_pdf:(i+1)*n_pdf,2] = post_pdf

        # create meshgrid
        Z = griddata((data_pts[:,0], data_pts[:,1]), data_pts[:,2], (X,Y), method='nearest')
        plt.contour(X,Y,Z, colors=color[c], levels=5)
        c_map = truncate_colormap(cmaps[c], 0.1, n=n_exp)
        plt.pcolormesh(X,Y,Z, cmap=c_map, shading='auto', alpha=alpha)

        c += 1
    
    plt.xlabel("Experiments")
    plt.ylabel(post_para_label)
    plt.colorbar(label="Probability density")
    #plt.legend()

    return fig

def plot_cum_post_2D_rvs(hist_diff, range, ylabel="p(A)-p(B)"):
    """_summary_

    Args:
        rvs_diff (_type_): _description_
        range (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    fig = plt.figure(figsize=FIGSIZE)

    n_bins = hist_diff.shape[1]
    x = np.arange(1, hist_diff.shape[0]+1)
    d = (range[0]-range[1])/n_bins
    bins = np.linspace(range[0]+d, range[1]+d, n_bins)
    X, Y = np.meshgrid(x,bins)

    # smooth data
    #data = scipy.ndimage.zoom(data, 3)
    
    plt.contour(X, Y, hist_diff.transpose(), levels=5, linewidths=0.5, colors='white')
    plt.pcolormesh(X,Y, hist_diff.transpose(), cmap="viridis", shading='auto')
    plt.xlabel("Experiments")
    plt.ylabel(ylabel)
    plt.colorbar(label="Probability density")

    return fig  

def plot_cum_post_1D_pdf(post, zip_post_para, labels, exp, model_para, post_para_label="Parameter"):
    """_summary_

    Args:
        post (_type_): _description_
        zip_post_para (_type_): _description_
        exp (_type_): _description_
        model_para (_type_): _description_
        post_para_label (str, optional): _description_. Defaults to "Parameter".

    Returns:
        _type_: _description_
    """
    
    fig = plt.figure(figsize=FIGSIZE)

    N_exp = len(exp)
    cmaps = [plt.cm.Reds(np.linspace(0.1, 1, N_exp)), 
             plt.cm.Blues(np.linspace(0.1, 1, N_exp)),]

    for z, cmap, lab in zip(zip_post_para, cmaps, labels):
        for i, post_para in enumerate(z):
            Post = post(*post_para)
            if i == N_exp-1:
                plt.plot(model_para, Post.pdf(model_para), color=cmap[i], label=lab)
            else:
                plt.plot(model_para, Post.pdf(model_para), color=cmap[i])

    plt.legend()
    plt.xlabel(post_para_label)
    plt.ylabel("Probability density")
    
    return fig

def plot_cum_post_1D_rvs(rvs, exp, model_para_pts, labels, post_para_label="Parameter"):
    """_summary_

    Args:
        rvs (_type_): _description_
        exp (_type_): _description_
        model_para_pts (_type_): _description_
        post_para_label (str, optional): _description_. Defaults to "Parameter".

    Returns:
        _type_: _description_
    """

    fig = plt.figure(figsize=FIGSIZE)

    n_exp = len(exp)
    cmaps = [plt.cm.Reds(np.linspace(0.1,1,n_exp)), 
             plt.cm.Blues(np.linspace(0.1,1,n_exp)),]

    for rv, cmap, lab in zip(rvs, cmaps, labels):
        for i,r in enumerate(rv):
            #plt.hist(r, bins=N_BINS, color=cmap[i], alpha=0.4, density=True)
            kde = gaussian_kde(r)
            plt.plot(model_para_pts, kde(model_para_pts), color=cmap[i])

    plt.xlabel(post_para_label)
    plt.ylabel("Probability density")

    return fig


def plot_anim_pdf(post, post_para, model_para_range, 
                model_para_label="Model parameter", list_hdi=[95,90,80,60], n_pdf=1000, interval=None, rvs=False):
    """
    Make an animation with the evolution of the posterior and the hdi's

    Args:
        post (scipy.stats): posterior distribution object (must have pdf, ppf, cdf methods)
        post_para (list): list with the cumulative posterior parameter for each 'experiment'
        model_para_i (float): lower limit for the model parameter
        model_para_f (float): upper limit for the model parameter
        model_para_label (str, optional): label for the model parameter. Defaults to "Model parameter".
        list_hdi (array_like, optional): list of desired hdi level to be displayed. Defaults to [95,90,80,60].
        n_pdf (int, optional): number of pts for the model parameter. Defaults to 1000.
        interval (float, optional): time in ms between each frame. 
                                    If None, this will be calculated so that the entire animation last 5s. Defaults to None.

    Returns:
        pyplot.animate.Funcanimation 
    """

    plt.rcParams["animation.html"] = "jshtml"
    model_para_i, model_para_f = model_para_range
    if interval is None:
        # default is 5s duration
        interval = 5000/len(post_para)

    n_exp = len(post_para)
    model_para_pts = np.linspace(model_para_i, model_para_f, n_pdf)

    # frame
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout()

    Post = post(*post_para[-1])

    # initialize posterior plot  
    axs[0].set_xlabel(model_para_label)
    axs[0].set_ylabel("Probabilty density")
    axs[0].set_xlim(model_para_i,model_para_f)
    ymax = np.max(Post.pdf(model_para_pts))
    axs[0].set_ylim(0, ymax)          
    anim_post, = axs[0].plot(model_para_pts, np.zeros_like(model_para_pts), color="black")
    anim_fill = axs[0].fill_between(model_para_pts, np.zeros_like(model_para_pts), alpha=0.5, color="b")

    # initialize hdi plot
    colors = plt.cm.viridis(np.linspace(0,1,4))
    axs[1].set_xlabel("Experiments")
    axs[1].set_ylabel(model_para_label)
    axs[1].set_ylim(model_para_i,model_para_f)
    # store fill_between obj in list
    anim_hdi = []
    x = np.arange(n_exp)
    y_up = np.zeros(n_exp)
    y_low = np.zeros(n_exp)
    for j,hdi in enumerate(list_hdi):
        anim_hdi.append(axs[1].fill_between(x, y_up, y_low,
                                            label="{}%".format(hdi), color=colors[j]))
    plt.legend()

    plt.close(fig)  # dont show initial plot

    # animation function  
    def animate(i):
        # ref: https://brushingupscience.com/2019/08/01/elaborate-matplotlib-animations/

        # update posterior plot
        ###############################

        Post = post(*post_para[i])
        post_pdf = Post.pdf(model_para_pts)
        anim_post.set_data(model_para_pts, post_pdf)

        # update fill
        # Get the paths created by fill_between
        path = anim_fill.get_paths()[0]
        # vertices is the part we need to change
        verts = path.vertices
        # Elements 1:Nx+1 encode the upper curve
        verts[1:n_pdf+1, 1] = post_pdf

        # update hdi plots
        ###############################

        for hdi_level,fill in zip(list_hdi, anim_hdi):
            
            # Get hdi
            hdi_up, hdi_low = helper.hdi(Post, level=hdi_level/100)

            if (hdi_low == np.nan) or (hdi_up == np.nan):
                hdi_low = 0
                hdi_up = np.max(post_pdf)

            # Get the paths created by fill_between
            path = fill.get_paths()[0]
            # vertices contain the x(1st dim) and y(2nd dim) pts
            verts = path.vertices
            # Elements 1:n_exp+1 encode the upper curve
            verts[i, 1] = hdi_up
            # Elements n_exp+2:-1 encode the lower curve, but
            # in right-to-left order
            verts[-i, 1] = hdi_low
            # It is unclear what 0th, Nx+1, and -1 elements
            # are for as they are not addressed here
        
        # set dynamic x and y limits
        axs[1].set_xlim(0, max(1, i-2))


    # call the animator.  blit=True means only re-draw the parts that have changed.
    return animation.FuncAnimation(fig, animate,
                            frames=n_exp, interval=interval, blit=False)


def plot_anim_rvs(rvs_list, model_para_range, 
                model_para_label="Model parameter", list_hdi=[95,90,80,60], 
                n_pts=N_PTS, interval=None, rvs=False):
    """
    Make an animation with the evolution of the posterior and the hdi's

    Args:
        post (scipy.stats): posterior distribution object (must have pdf, ppf, cdf methods)
        post_para (list): list with the cumulative posterior parameter for each 'experiment'
        model_para_i (float): lower limit for the model parameter
        model_para_f (float): upper limit for the model parameter
        model_para_label (str, optional): label for the model parameter. Defaults to "Model parameter".
        list_hdi (array_like, optional): list of desired hdi level to be displayed. Defaults to [95,90,80,60].
        n_pdf (int, optional): number of pts for the model parameter. Defaults to 1000.
        interval (float, optional): time in ms between each frame. 
                                    If None, this will be calculated so that the entire animation last 5s. Defaults to None.

    Returns:
        pyplot.animate.Funcanimation 
    """

    plt.rcParams["animation.html"] = "jshtml"

    if interval is None:
        # default is 5s duration
        interval = 5000/len(rvs_list)

    n_exp = len(rvs_list)

    if model_para_range is None:
        model_para_i = min(rvs_list[0])
        model_para_f = max(rvs_list[0])
    else:
        model_para_i, model_para_f = model_para_range
    model_para_pts = np.linspace(model_para_i, model_para_f, n_pts)

    # frame
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout()

    Post = rvs_list[0]

    # initialize posterior plot  
    axs[0].set_xlabel(model_para_label)
    axs[0].set_ylabel("Probabilty density")
    axs[0].set_xlim(min(Post),max(Post))
    ymax = max(np.histogram(rvs, bins=N_BINS)[0])
    axs[0].set_ylim(0, ymax) 
    anim_post, = axs[0].plot(model_para_pts, np.zeros_like(model_para_pts), color="black")
    anim_fill = axs[0].fill_between(model_para_pts, np.zeros_like(model_para_pts), alpha=0.5, color="b")

    # initialize hdi plot
    colors = plt.cm.viridis(np.linspace(0,1,4))
    axs[1].set_xlabel("Experiments")
    axs[1].set_ylabel(model_para_label)
    axs[1].set_ylim(model_para_i,model_para_f)
    # store fill_between obj in list
    anim_hdi = []
    x = np.arange(n_exp)
    y_up = np.zeros(n_exp)
    y_low = np.zeros(n_exp)
    for j,hdi in enumerate(list_hdi):
        anim_hdi.append(axs[1].fill_between(x, y_up, y_low,
                                            label="{}%".format(hdi), color=colors[j]))
    plt.legend()

    plt.close(fig)  # dont show initial plot

    # animation function  
    def animate(i):
        # ref: https://brushingupscience.com/2019/08/01/elaborate-matplotlib-animations/

        # update posterior plot
        ###############################

        kde = gaussian_kde(rvs_list[i])
        Post = helper.KDE(kde, min(rvs_list[i]), max(rvs_list[i]), N_PTS)
        post_pdf = kde(model_para_pts)
        anim_post.set_data(model_para_pts, post_pdf)

        # update fill
        # Get the paths created by fill_between
        path = anim_fill.get_paths()[0]
        # vertices is the part we need to change
        verts = path.vertices
        # Elements 1:Nx+1 encode the upper curve
        verts[1:n_pts+1, 1] = post_pdf

        # update hdi plots
        ###############################


        for hdi_level,fill in zip(list_hdi, anim_hdi):
            
            # Get hdi
            hdi_up, hdi_low = helper.hdi(Post, level=hdi_level/100)

            if (hdi_low == np.nan) or (hdi_up == np.nan):
                hdi_low = 0
                hdi_up = np.max(post_pdf)

            # Get the paths created by fill_between
            path = fill.get_paths()[0]
            # vertices contain the x(1st dim) and y(2nd dim) pts
            verts = path.vertices
            # Elements 1:n_exp+1 encode the upper curve
            verts[i, 1] = hdi_up
            # Elements n_exp+2:-1 encode the lower curve, but
            # in right-to-left order
            verts[-i, 1] = hdi_low
            # It is unclear what 0th, Nx+1, and -1 elements
            # are for as they are not addressed here
        
        # set dynamic x and y limits
        axs[1].set_xlim(0, max(1, i-2))


    # call the animator.  blit=True means only re-draw the parts that have changed.
    return animation.FuncAnimation(fig, animate,
                            frames=n_exp, interval=interval, blit=False)


def make_plot_tot(make_rvs, make_pdf, group, xlabel, n_rvs=N_SAMPLE, para_range=None, n_pts=N_PTS):
    """
    plot the posterior distribution for the total result for all conjugated prior classes

    Args:
        n_rvs (int, optional): number of random value for the histogram. Defaults to 1000.
        prange (list, optional): [lower, upper] limit for p. Defaults to None.
    """       
        
    if (group == "A") or (group == "B"):
        rvs = make_rvs(group=group, N_sample=n_rvs)
        if para_range is None:
            para_range = [np.min(rvs), np.max(rvs)]
        model_para_pts, post = make_pdf(group=group, para_range=para_range)
        fig = plot_tot([rvs], model_para_pts, [post], labels=[group], xlabel=xlabel)
    
    elif group == "diff":
        rvs_A = make_rvs(group="A", N_sample=n_rvs) 
        rvs_B = make_rvs(group="B", N_sample=n_rvs)
        rvs_diff = rvs_A-rvs_B
        if para_range is None:
            para_range = [np.min(rvs_diff), np.max(rvs_diff)]
        model_para_pts = np.linspace(para_range[0],para_range[1],n_pts)
        fig = plot_tot([rvs_diff],model_para_pts, labels=["A-B"], xlabel="Difference in "+xlabel)
    
    elif group == "AB":
        rvs_A = make_rvs(group="A", N_sample=n_rvs)
        rvs_B = make_rvs(group="B", N_sample=n_rvs)
        rvs_tmp = np.concatenate((rvs_A, rvs_B))
        if para_range is None:
            para_range = [np.min(rvs_tmp), max(rvs_tmp)]
        model_para_pts, post_A = make_pdf(group="A", para_range=para_range)
        _, post_B = make_pdf(group="B", para_range=para_range)
        fig = plot_tot([rvs_A, rvs_B], model_para_pts, [post_A, post_B],
                                        labels=["A", "B"], 
                                        xlabel=xlabel)    
    else:
        raise SyntaxError("group can only be A,B,diff or AB")

    return fig

def plot_helper(make_rvs, make_cum_post_para, conjugated_prior, 
                group, type, N_exp, 
                n_pdf, n_rvs,
                label1, label2,
                xrange=None):
    """
    Wrapper function for the different plots

    Args:
        make_rvs (_type_): _description_
        make_cum_post_para (_type_): _description_
        conjugated_prior (_type_): _description_
        group (_type_): _description_
        N_exp (_type_): _description_
        n_pdf (_type_): _description_
        n_rvs (_type_): _description_
        label1 (_type_): _description_
        label2 (_type_): _description_
        xrange (_type_): _description_

    Returns:
        _type_: _description_
    """

    experiment_cnt = np.arange(1, N_exp+1)

    # plot posterior for one group
    if (group == "A") or (group == "B"):

        # values for x axis
        # if range of probabilities is not given, take it from the rvs 
        if xrange is None:
            rvs = make_rvs(group=group)
            xrange = [np.min(rvs), np.max(rvs)] 
        x_pts = np.linspace(xrange[0],xrange[1],n_pdf)

        # make a list of cumulative 
        zip_post_para = [zip(*make_cum_post_para(group=group))]
        labels = [group]

        # 2D map plot
        if type == "2D":
            fig = plot_cum_post_2D_pdf(conjugated_prior, zip_post_para, 
                                                    labels, 
                                                    experiment_cnt, x_pts, 
                                                    post_para_label=label1)
        
        # 1D plot
        elif type == "1D":
            fig = plot_cum_post_1D_pdf(conjugated_prior, zip_post_para, 
                                                        labels, 
                                                        experiment_cnt, x_pts, 
                                                        post_para_label=label1)
    
    # plot posterior for both groups
    elif group == "AB":

        zip_post_para = [zip(*make_cum_post_para(group="A")), 
                            zip(*make_cum_post_para(group="B"))]
        labels = ["A", "B"]

        if xrange is None:
            rvs = np.concatenate((make_rvs(), make_rvs(group="B")))
            xrange = [np.min(rvs), np.max(rvs)]
        x_pts = np.linspace(xrange[0],xrange[1],n_pdf)

        if type == "2D":
            fig = plot_cum_post_2D_pdf(conjugated_prior, zip_post_para, 
                                                    labels, 
                                                    experiment_cnt, x_pts, 
                                                    post_para_label=label1)
        elif type == "1D":
            fig = plot_cum_post_1D_pdf(conjugated_prior, zip_post_para, 
                                                    labels, 
                                                    experiment_cnt, x_pts, 
                                                    post_para_label=label1)
    
    # plot posterior for the difference
    elif group == "diff":

        # list of parameters for the gamma function
        A_a, A_b = make_cum_post_para(group="A")
        B_a, B_b = make_cum_post_para(group="B")
        # rvs for the differences after the first "experiment"
        rvs_diff = conjugated_prior(a=A_a[0], b=A_b[0]).rvs(size=n_rvs)\
                    -conjugated_prior(a=B_a[0], b=B_b[0]).rvs(size=n_rvs)
        # x range
        if xrange is None:
            xrange = [np.min(rvs_diff), max(rvs_diff)]
        x_pts = np.linspace(xrange[0],xrange[1],n_pdf)

        # build list of histogram
        hist_list = []
        rvs_list = []
        for aa,ab,ba,bb in zip(A_a, A_b, B_a, B_b):
            rvs = conjugated_prior(a=aa, b=ab).rvs(size=n_rvs)\
                    -conjugated_prior(a=ba, b=bb).rvs(size=n_rvs)
            hist = np.histogram(rvs, bins=N_BINS, range=xrange, density=True)[0]
            hist_list.append(hist)
            rvs_list.append(rvs)

        if type == "2D":
            hist_arr = np.array(hist_list)
            fig = plot_cum_post_2D_rvs(hist_arr, xrange,
                                        ylabel=r"${}$(A)-${}$(B)".format(label2, label2))
            
        elif type == "1D":
            fig = plot_cum_post_1D_rvs([rvs_list], experiment_cnt, 
                                        model_para_pts=x_pts, labels=["A-B"],
                                        post_para_label="Difference of "+label1)
        
    else:
        raise ValueError("group must be either 'A', 'B', 'diff' or 'AB'")

    return fig
