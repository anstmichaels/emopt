"""Various functions associated loading and saving files.
"""

from misc import run_on_master, warning_message, NOT_PARALLEL, COMM
from grid import Polygon
import numpy as np

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "0.4"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

@run_on_master
def plot_iteration(field, structure, W, H, foms, fname='', layout='auto',
                   show_now=False, dark=True, Nmats=2):
    """Plot the current iteration of an optimization.

    Plots the following:
        1) A 2D field
        2) A 2D representation of the geometry
        3) A line plot of the figure of merit history

    When plotting 3D structures, multiple 2D slices of the geometry can be
    passed in by passing a 3D array to structure. The different "pages" of this
    array will be flattened into a 2D representation of the sliced geometry.

    Furthermore, multiple figures of merit may be plotted. This is done by
    putting more than one key:value pair in the foms dictionary. The key names
    will be used as the legend label.

    Parameters
    ----------
    field : numpy.ndarray
        2D array containing current iteration's field slice
    structure : numpy.ndarray
        Either a 2D array containing a representation of the current structure
        or a 3D array containing a FEW 2D slices.
    W : float
        The width of the field/structure
    H : float
        The height of the field/structure
    foms : dict
        A dictionary containing the fom history. The key strings should be
        names which describe each supplied figure of merit
    fname : str
        Filename for saving
    layout : str
        Layout method. 'auto' = automatically choose layout based on aspect
        ratio. 'horizontal' = single row layour. 'vertical' = single column
        layout. 'balanced' = field+structure on left, foms on right.
    show_now : bool
        If True, show plot now (warning: this is a blocking operation)
    dark : bool
        If True, use a dark color scheme for plot. (default = True)

    Returns
    -------
    matplotlib.pyplot.figure
        The current matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    dpi=300
    aspect = W/H

    # determine and setup the plot layout
    unknown_layout = layout not in ['auto', 'horizontal', 'vertical', 'balanced']
    if(unknown_layout):
        warning_message("Unknown layout '%s'. Using 'auto'" % (layout), 'emopt.io')

    if(layout == 'auto' or unknown_layout):
        if(aspect < 1.0):
            layout = 'horizontal'
        elif(aspect >= 1.0 and aspect <= 2.5):
            layout = 'balanced'
        else:
            layout = 'vertical'

    gs = None
    Wplot = 10.0
    if(layout == 'horizontal'):
        f = plt.figure(figsize=(Wplot*aspect*3, Wplot))
        gs = gridspec.GridSpec(1,3, height_ratios=[1])
        ax_field = f.add_subplot(gs[:,0])
        ax_struct = f.add_subplot(gs[:,1])
        ax_foms = f.add_subplot(gs[:,2])
    elif(layout == 'vertical'):
        f = plt.figure(figsize=(Wplot, Wplot/aspect*4))
        gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1])
        ax_field = f.add_subplot(gs[0,:])
        ax_struct = f.add_subplot(gs[1,:])
        ax_foms = f.add_subplot(gs[2, :])
    elif(layout == 'balanced'):
        f = plt.figure(figsize=(Wplot, Wplot/aspect*2/1.5))
        gs = gridspec.GridSpec(2,3, height_ratios=[1,1])
        ax_field = f.add_subplot(gs[0,0:2])
        ax_struct = f.add_subplot(gs[1,0:2])
        ax_foms = f.add_subplot(gs[:,2])

    # define dark colormaps
    if(dark):
        field_cols=['#3d9aff', '#111111', '#ff3d63']
        field_cmap=LinearSegmentedColormap.from_list('field_cmap', field_cols)

        struct_cols=['#212730', '#bcccdb']
        struct_cmap=LinearSegmentedColormap.from_list('struct_cmap', struct_cols)
    else:
        field_cmap = 'seismic'
        struct_cmap = 'Blues'

    extent = [0, W, 0, H]

    fmax = np.max(field)
    fmin = np.min(field)

    if(fmax > 0 and fmin < 0):
        fmin = -1*fmax # symmetrize
    if(fmax < 0 and fmin < 0):
        fmax = 0
        field_cmap = 'hot_r'
    if(fmax > 0 and fmin > 0):
        fmin = 0
        field_cmap = 'hot'

    ax_field.imshow(field, extent=extent, vmin=fmin, vmax=fmax, cmap=field_cmap)


    # "flatten" multi-layer structure
    if(len(structure.shape) == 3):
        Nlevels = structure.shape[0] + 1
        structure = np.sum(structure, axis=0) / structure.shape[0]
    else:
        Nlevels = Nmats

    smin = np.min(structure)
    smax = np.max(structure)
    ax_struct.imshow(structure, extent=extent, vmin=smin, vmax=smax,
                     cmap=struct_cmap)

    # outline structure in field plot
    ax_field.contour(np.flipud(structure), extent=extent, levels=Nlevels,
                      colors='#666666', linewidths=0.1)

    # set dark colors
    # plot title with important iteration number, etc
    # sum along z-axis for structure

    # define fom plot colors
    Nplot = len(foms.keys())
    red = np.linspace(0.2, 1.0, Nplot)
    blue = np.linspace(1.0, 0.2, Nplot)
    green = np.zeros(Nplot)
    red_base = 0.0
    blue_base = 0.0
    green_base = 0.55

    i = 0
    Niter = 0
    current_foms = []
    for desc in foms.keys():
        fom = foms[desc]
        Niter = len(fom)
        iters = np.arange(Niter)
        current_foms.append(fom[-1])

        pcolor = (red_base + red[i], green_base +green[i], blue_base + blue[i])
        i += 1
        pline = ax_foms.plot(iters, fom, '.-', color=pcolor, markersize=10)

    ax_foms.set_xlabel('Iteration', fontsize=12)
    ax_foms.set_ylabel('Figure of Merit', fontsize=12)
    ax_foms.legend(foms.keys(), loc=4)
    ax_foms.grid(True, linewidth=0.5)

    # general tick properties
    for ax in [ax_field, ax_struct, ax_foms]:
        ax.get_yaxis().set_tick_params(which='both', direction='in', top=True,
                                      right=True)
        ax.get_xaxis().set_tick_params(which='both', direction='in', top=True,
                                      right=True)

    # Dark theme easier on eyes
    if(dark):
        c_text_main = '#BBBBBB'
        c_bg_main = '#101010'
        c_plot_bg = '#353535'
        c_lines = '#666666'
        c_plot_tick = '#CCCCCC'
        c_plot_grid = '#555555'
        f.patch.set_facecolor(c_bg_main)
        for ax in [ax_field, ax_struct, ax_foms]:
            for tl in ax.get_xticklabels():
                tl.set_color(c_text_main)
            for tl in ax.get_yticklabels():
                tl.set_color(c_text_main)
            ax.xaxis.get_label().set_color(c_text_main)
            ax.yaxis.get_label().set_color(c_text_main)

            for spine in ax.spines:
                ax.spines[spine].set_color(c_lines)

            ax.get_yaxis().set_tick_params(color=c_plot_tick)
            ax.get_xaxis().set_tick_params(color=c_plot_tick)

        ax_foms.set_facecolor(c_plot_bg)
        ax_foms.grid(True, color=c_plot_grid, linewidth=0.5)
    else:
        c_text_main = '#000000'

    # title contains info of current iteration
    f.suptitle(''.join(['Iteration %d, ' % (Niter),
                     'FOMs = '] + ['%0.4f  ' % (fom) for fom in current_foms]),
            fontsize=12, color=c_text_main)

    if(fname != ''):
        if(dark):
            plt.savefig(fname, dpi=300, facecolor=c_bg_main, bbox_inches='tight')
        else:
            plt.savefig(fname, dpi=300, bbox_inches='tight')

    if(show_now):
        plt.tight_layout()
        plt.show()

@run_on_master
def save_results(fname, data, additional=None):
    """Save an hdf5 file containing common simulation and optimization results.

    The input to this function is a fileneam and a dictionary which
    contains the following possible items:
    W - Width of simulation
    H - Height of simulation
    dx - x grid spacing
    dy - y grid spacing
    M - number of rows in field matrices
    N - number of columns in field matrices
    w_pml_x - PML width in x
    w_pml_y - PML height in y
    Ex - x component of electric field
    Ey - y component of electric field
    Ez - z component of electric field
    Hx - x component of magnetic field
    Hy - y component of magnetic field
    Hz - z component of magnetic field
    eps - The permittivity of the system
    mu - The permeability of the system
    params - The design parameters of the system
    foms - List of figure of merits achieved during optimization

    A second optional dict can be passed as well which contains additional data to store
    that is not recognized as a typical simulation or optimization result

    Notes
    -----
    This function depends on h5py

    Parameters
    ----------
    fname : string
        The name and path of file which will be saved (Note: a file extention is added automatically)
    data : dict
        The simulation and optimization results to be saved
    additional : dict
        Any addtional data to save
    """
    import h5py

    fname_full = ''.join([fname, '.h5'])
    with h5py.File(fname_full, "w") as hf:
        group_sim = hf.create_group("simulation")
        group_opt = hf.create_group("optimization")
        group_misc = hf.create_group("misc")

        # simulation attributes
        if 'W' in data:
            group_sim.attrs['W'] = data['W']
        if 'H' in data:
            group_sim.attrs['H'] = data['H']
        if 'X' in data:
            group_sim.attrs['X'] = data['X']
        if 'Y' in data:
            group_sim.attrs['Y'] = data['Y']
        if 'Z' in data:
            group_sim.attrs['Z'] = data['Z']
        if 'dx' in data:
            group_sim.attrs['dx'] = data['dx']
        if 'dy' in data:
            group_sim.attrs['dy'] = data['dy']
        if 'dz' in data:
            group_sim.attrs['dz'] = data['dz']
        if 'M' in data:
            group_sim.attrs['M'] = data['M']
        if 'N' in data:
            group_sim.attrs['N'] = data['M']
        if 'w_pml_x' in data:
            group_sim.attrs['w_pml_x'] = data['w_pml_x']
        if 'w_pml_y' in data:
            group_sim.attrs['w_pml_y'] = data['w_pml_y']

        # Simulation results
        if 'Ex' in data:
            group_sim.create_dataset('Ex', data=data['Ex'])
        if 'Ey' in data:
            group_sim.create_dataset('Ey', data=data['Ey'])
        if 'Ez' in data:
            group_sim.create_dataset('Ez', data=data['Ez'])
        if 'Hx' in data:
            group_sim.create_dataset('Hx', data=data['Hx'])
        if 'Hy' in data:
            group_sim.create_dataset('Hy', data=data['Hy'])
        if 'Hz' in data:
            group_sim.create_dataset('Hz', data=data['Hz'])
        if 'eps' in data:
            group_sim.create_dataset('eps', data=data['eps'])
        if 'mu' in data:
            group_sim.create_dataset('mu', data=data['mu'])

        # Optimization results
        if 'params' in data:
            group_opt.create_dataset('params', data=data['params'])
        if 'foms' in data:
            group_opt.create_dataset('foms', data=data['foms'])

        # any additional data
        if(additional is not None):
            for key in additional:
                group_misc.create_dataset(key, data=additional[key])


def load_results(fname, bcast=True):
    """
    Load data that has been saved with the :func:`save_results` function.

    Parameters
    ----------
    fname : string
        The file name and path of file from which data is loaded.
    bcast : bool (optional)
        If True, broadcast the loaded data to all of the processes. (default =
        True)

    Returns
    -------
    dict
        A dictionary containing the loaded data.
    """
    import h5py

    data = {}

    if(NOT_PARALLEL):
        fname_full = ''.join([fname, '.h5'])
        with h5py.File(fname_full, "r") as fh5:

            for key in fh5['simulation'].keys():
                data[key] = fh5['simulation'][key][...]

            for key in fh5['simulation'].attrs.keys():
                data[key] = fh5['simulation'].attrs[key][...]

            for key in fh5['optimization'].keys():
                data[key] = fh5['optimization'][key][...]

            for key in fh5['misc'].keys():
                data[key] = fh5['misc'][key][...]

    if(bcast):
        data = COMM.bcast(data, root=0)

    return data

def load_gds_txt(fname, unit_mult=1e-3):
    """Load a very simple text-format GDS file exported from klayout.

    Notes
    -----
    This function is not really optimized, so don't use it for very large GDS
    files. It is really only meant for simpler structures (i.e. not entire
    complicated layouts).

    Parameters
    ----------
    fname : str
        The path/name of the GDS file to load.
    unit_mult : float
        The multiplicative factor which yields numbers in the desired
        unit system. For example, if you do everything in um but klayout
        exports in nm, then unit_mult=1e-3

    Returns
    -------
    dict
        A dictionary containing lists of polygons. The keys of the dictionary are
        the layer numbers used in klayout. The values of the dictionary are
        lists containing the shapes that make up each layer.
    """

    # load contents
    contents = ''
    with open(fname, 'r') as fin:
        contents = fin.read()

    # clean up klayout weirdness
    contents = contents.replace('XY ', 'XY\n')

    # break up lines
    lines = contents.split('\n')
    lines = [line.strip() for line in lines]

    # find blocks of boundaries
    iB = [i for i,x in enumerate(lines) if x == 'BOUNDARY']
    iE = [i for i,x in enumerate(lines) if x == 'ENDEL']

    # loop through all boundaries and create Polygons :)
    polys = []
    layers = []
    for i,j in zip(iB, iE):
        layer_str = lines[i+1]
        layer = float(layer_str.split(' ')[1])
        layers.append(layer)

        xs = []
        ys = []
        for k in range(i+4, j):
            coordinate = lines[k]
            xy = coordinate.split(': ')
            xs.append(float(xy[0])*unit_mult)
            ys.append(float(xy[1])*unit_mult)

        p = Polygon()
        p.set_points(xs, ys)
        polys.append(p)

    # store everythin in a dictionary
    polygons = {}
    for p,l in zip(polys, layers):
        if(l in polygons):
            polygons[l].append(p)
        else:
            polygons[l] = [p]

    return polygons
