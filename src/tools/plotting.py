"""
plotting.py
This file contains miscillaneous functions used for plotting

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.coordinates import SkyCoord
import astropy.units as u

mpl.style.use('seaborn-v0_8-bright')

mpl.rc({
    'font.family': 'san-serif',
    "font.sans-serif": 'Computer Modern Sans Serif',
    "font.weight": 'medium',
    'font.size'   : 16,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'text.usetex': False,
    'axes.labelsize': 'large',
    'axes.formatter.use_mathtext': True,
    'xtick.labelsize': 48,
    'axes.labelsize': 'large',
    'axes.edgecolor': 'k',
    'axes.facecolor': 'k',
    'mathtext.fontset': 'Computer Modern Sans Serif',
    'mathtext.rm': 'Computer Modern Sans Serif',
    'mathtext.sf': 'Computer Modern Sans Serif',
    'mathtext.tt': 'Computer Modern Sans Serif'
})

mpl.rcParams['agg.path.chunksize'] = 100000
mpl.style.reload_library()
plt.rcParams['font.family'] = 'sans-serif'

def set_labels(ax, xlabel, ylabel, title='', xlim='default', ylim='default',\
 legend=False, format_ticks=True, set_as_white=True, log_scale=[False, 'xy'], labelpad=[1, 1, 1], markerscale=1,
 legend_outside_plot :bool = False, legend_loc: str = 'best', bbox_to_anchor=(0.960, 1.018), title_fontsz = 24, ncol=1):
    """
    Function defining plot properties
    @param ax :: axes to be held
    @param xlabel, ylabel :: labels of the x-y axis
    @param title :: title of the plot
    @param xlim, ylim :: x-y limits for the axis
    """
    ax.set_xlabel(xlabel, fontsize=24, labelpad=labelpad[0])
    ax.set_ylabel(ylabel, fontsize=24, labelpad=labelpad[1])

    if xlim != 'default':
        ax.set_xlim(xlim)

    if ylim != 'default':
        ax.set_ylim(ylim)

    if legend:
        ax.legend(loc=legend_loc, fontsize=14, frameon=True, markerscale=markerscale, facecolor='w', ncol=ncol)

    if legend_outside_plot:
        ax.legend(bbox_to_anchor=bbox_to_anchor, fancybox=False, framealpha=1, frameon=True, edgecolor='k', ncol=ncol)

    if format_ticks:
        ax.tick_params(axis='x', which='both',  direction='in', length=6, width=1)
        ax.tick_params(axis='y', which='both',  direction='in', length=6, width=1)

    if set_as_white:
        color='k'
    else:
        color='w'
    ax.set_title(title, fontsize=title_fontsz, color=color)
    ax.grid(False)

    if log_scale[0]:
        if log_scale[1] == 'x':
            ax.set_xscale('log')
        if log_scale[1] == 'y':
            ax.set_yscale('log')
        if log_scale[1] == 'xy':
            ax.set_xscale('log')
            ax.set_yscale('log')
    return

def plot_3D(input_arr, step: int = 100, s: float =1e-1, alpha: float =0.7, plot_path: str = None, plotname: str = None, xlim=None, ylim=None, zlim=None,
            figsize=(10, 10), labelpad=[100, 20, 20]):
    """
    input_arr :: (N, 3)
        in units ckpc/h
    step :: int
        the step size for skipping points
    s :: float
        the marker size for representing the particles
    alpha :: float  
        the transperancy of the particles (matplotlib kwargs)
    plot_path :: str
        directory to save the plot
    plotname :: str
        name of the plot
    xlim, ylim, zlim ::
        the x, y, z limits of the axis
    figsize :: 
        the size of the figure
    """
    if plot_path is None:
        raise ValueError(f"{plot_path=}, you need to input directory where you want to save the plot")
    if plotname is None:
        plotname = "test.png"
    fig = plt.figure(figsize=figsize)
    
    ax = plt.axes(projection='3d')

    c_list = []
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(input_arr[::step, 0]))))
    for i in range(len(input_arr[::step, 0])):
        c_list.append(next(color))
    ax.scatter3D(input_arr[::step, 0], input_arr[::step, 1], input_arr[::step, 2], c=c_list, s=s, alpha=alpha)

    set_labels(ax, 'X [302.6 Mpc]', 'Y', labelpad=labelpad)
    
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)

    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_zlabel('\n Z', labelpad=labelpad[2])

    fig.savefig(f"{plot_path}/{plotname}", facecolor='w')
    return ax, fig

def set_polar_labels(ax, rlabel, theta_label, theta_lim= [0, 180], rmax: float = None):
    """Function for labelling polar plots """
    ax.set_thetamin(theta_lim[0])
    ax.set_thetamax(theta_lim[1])
    if rmax is not None:
        ax.set_rmax(rmax)

    ax.set_xlabel(rlabel, )
    ax.set_ylabel(theta_label, rotation=6)
    
    ax.grid(True)
    return ax

def ra_dec_2_theta_phi(ra,dec):
    """
    Function to convert the ra and dec into theta and phi

    Parameters
    ----------
    ra :: arr, float
        The array with the ra of the points
    dec :: arr, float
        The array with the dec of the points
    """
    c = SkyCoord(ra=ra, dec=dec, frame='icrs', unit=u.deg)
    phi, theta = c.ra.wrap_at(180*u.deg).radian, 0.5 * np.pi - c.dec.radian
    return theta, phi