
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as colors

import numpy as np
import scipy.interpolate as interpolate
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import scipy.ndimage

import src.helper as helper

FIGSIZE=(6,4)
COLORS = ["red", "blue", "green", "orange"]
N_BINS = 20
N_SAMPLE = 5000
N_PTS = 2000

def plot_tot(rvs, model_para_pts, pdf=None, xlabel="Model parameter", labels=[None]):
    """
    Plot the probabilty density and histogram for a posterior distribution

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
    plt.ylabel("Probabilty density")
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
    plt.colorbar(label="Probabilty density")
    #plt.legend()

    return fig

def plot_cum_post_2D_rvs(hist_diff, range, ylabel="p(A)-p(B) (%)"):
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
    plt.colorbar(label="Probabilty density")

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

    n_exp = len(exp)
    cmaps = [plt.cm.Reds(np.linspace(0.1,1,n_exp)), 
             plt.cm.Blues(np.linspace(0.1,1,n_exp)),]

    for z, cmap, lab in zip(zip_post_para, cmaps, labels):
        for i, post_para in enumerate(z):
            Post = post(*post_para)
            if i == n_exp-1:
                plt.plot(model_para, Post.pdf(model_para), color=cmap[i], label=lab)
            else:
                plt.plot(model_para, Post.pdf(model_para), color=cmap[i])

    plt.legend()
    plt.xlabel(post_para_label)
    plt.ylabel("Probabilty density")
    
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
    plt.ylabel("Probabilty density")

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
                hdi_low = hdi_up = 0

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
                hdi_low = hdi_up = 0

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