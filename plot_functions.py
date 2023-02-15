
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import scipy.interpolate as interpolate

import helper

def plot_tot(rvs, pdf, model_para_pts, xlabel="Model parameter"):
    """
    Plot the probabilty density and histogram for a posterior distribution

    Args:
        rvs (np.array): array with the random values for the histogramm.
        pdf (np.array): array with axis 0 the parameter value and axis 1 the probability density.
        xlabel (strin, optional): label for the x axis (model parameter)
    """

    #plot rvs
    if rvs is not None:
        plt.hist(rvs, density=True, alpha=0.4, label="rvs from {} pts".format(len(rvs)))
    # plot pdf
    plt.plot(model_para_pts, pdf, label="pdf", color="black")

    plt.xlabel(xlabel)
    plt.ylabel("Probabilty density")
    plt.legend()
    plt.show()


def plot_cum_post(post, zip_post_para, exp, model_para, n_pdf, post_para_label="Parameter", type="2D"):
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

    n_exp = len(exp)

    if type == "2D":
        data_pts = np.zeros((n_pdf*n_exp,3))

        for i,post_para in enumerate(zip_post_para):
            Post = post(*post_para)
            post_pdf = Post.pdf(model_para)
            data_pts[i*n_pdf:(i+1)*n_pdf,0] = np.ones(n_pdf)*exp[i]
            data_pts[i*n_pdf:(i+1)*n_pdf,1] = model_para
            data_pts[i*n_pdf:(i+1)*n_pdf,2] = post_pdf

        # create meshgrid
        X,Y =  np.meshgrid(exp, model_para)
        Z = interpolate.griddata((data_pts[:,0], data_pts[:,1]), data_pts[:,2], (X,Y), method='nearest')

        plt.pcolormesh(X,Y,Z, cmap="Blues", shading='auto')
        plt.xlabel("Experiments")
        plt.ylabel(post_para_label)
        plt.colorbar(label="Probabilty density")
        plt.show()
    
    elif type == "1D":
        
        colors = plt.cm.viridis(np.linspace(0,1,n_exp))

        for i,post_para in enumerate(zip_post_para):
            Post = post(*post_para)
            plt.plot(model_para, Post.pdf(model_para), color=colors[i])

        plt.xlabel(post_para_label)
        plt.ylabel("Probabilty density")
        plt.show()

def plot_anim(post, post_para, model_para_i, model_para_f, 
                model_para_label="Model parameter", list_hdi=[95,90,80,60], n_pdf=1000, interval=None):
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