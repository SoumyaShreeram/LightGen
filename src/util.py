"""
util.py contains all the helper functions

Author: Soumya Shreeram
Date created: 21th Feb 2023
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=0.6774*100, Om0=0.3089, Tcmb0=2.7255, Ob0=0.0486, Neff=3.046)
from astropy.cosmology import z_at_value
from scipy.interpolate import CubicSpline
import astropy.units as u
from astropy.table import vstack, QTable
from astropy.coordinates import SkyCoord

import os 
import logging
logger = logging.getLogger(__name__)

# local imports
from tools import remap as re


def get_observer_coordinates(u, outfile_path: str = None) -> None:
    """Function to get the location of the observer in the remapped coordinates
    Parameters
    ----------
    u :: (u1, u2, u3) Tuple 
        3x3 transformation matrix
    outfile_path :: str
        the location where the observer locations are saved
    """
    # coordinates of the edges in original reference frame
    edges = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
    
    u1, u2, u3 = u[0], u[1], u[2]
    C = re.Cuboid(u1, u2, u3)
    print('Transformation matrix: ', u1, u2, u3)
    
    # obtain the transformed coordinated
    edges_t = np.zeros(np.shape(edges))
    X_t, Y_t, Z_t = np.zeros(len(edges_t[:,0])), np.zeros(len(edges_t[:,0])), np.zeros(len(edges_t[:,0]))
    for i, (x, y, z) in enumerate(zip(edges[:,0], edges[:,1], edges[:,2])):
        print('old ', x, y, z)
        X_t[i], Y_t[i], Z_t[i] = C.InverseTransform(x, y, z)
        print('new ', X_t[i], Y_t[i], Z_t[i])
    edges_t[:,0], edges_t[:,1], edges_t[:,2] = X_t, Y_t, Z_t
    np.save(outfile_path, np.array([edges, edges_t]), allow_pickle=True)
    return 

def get_edges2D(arrX, arrY, return_minmax: bool = False) -> np.array:
    """Function to get the edges of a parallelopiped
    Parameters
    ----------
    arr :: 3D array with dimensions (m, 3)
        the input parallelopiped whose verticies are to be determined

    Return 
    ------
    edges :: 1D array
        the verticies of the input 3D data
    """
    x_min, x_max = np.min(arrX), np.max(arrX)
    y_min, y_max = np.min(arrY), np.max(arrY)
    
    dict_minmax = {'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max}
    edges = []
    for x in[x_min, x_max]:
        for y in [y_min, y_max]:
            edges.append([x, y])
    if return_minmax:
        return np.array(edges), dict_minmax
    else:
        return np.array(edges)
    
def get_edges(arr, return_minmax: bool = False) -> np.array:
    """Function to get the edges of a parallelopiped
    Parameters
    ----------
    arr :: 3D array with dimensions (m, 3)
        the input parallelopiped whose verticies are to be determined

    Return 
    ------
    edges :: 1D array
        the verticies of the input 3D data
    """
    x_min, x_max = np.min(arr[:, 0]), np.max(arr[:, 0])
    y_min, y_max = np.min(arr[:, 1]), np.max(arr[:, 1])
    z_min, z_max = np.min(arr[:, 2]), np.max(arr[:, 2])

    dict_minmax = {'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max,}
    edges = []
    for x in[x_min, x_max]:
        for y in [y_min, y_max]:
            for z in [z_min, z_max]:
                edges.append([x, y, z])
    if return_minmax:
        return np.array(edges), dict_minmax
    else:
        return np.array(edges)
    

def form_verticies(L, B, H):
    "Function to get the verticies of a cuboid given its length, L, breadth, B, and height, H."
    vertex = []
    for l in [0, L]:
        for b in [0, B]:
            for h in [0, H]:
                vertex.append((l, b, h))
    return np.array(vertex)


def get_angle(vec1, vec2):
    "Returns the angle between two vectors"
    cos_theta = np.dot(vec1, vec2)/ (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    theta = np.arccos(cos_theta)
    return np.rad2deg(theta)

def area_rectangle_on_sphere(theta, phi):
    "Area on a sphere given the angular values at fixed radius"
    theta, phi = np.deg2rad(theta), np.deg2rad(phi)
    area = 4*np.arcsin(np.tan(theta/2)*np.tan(phi/2))*(180/np.pi)**2
    return area

def cartesian2polar(x, y, z):
    """Function to convert cartesian coordinates to polar coordinates"""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y/x)
    phi = np.arcsin(z/r)
    return r, theta, phi

def polar2cartesian(r, theta, phi):
    x = r*np.cos(phi)*np.cos(theta)
    y = r*np.cos(phi)*np.sin(theta)
    z = r*np.sin(phi)
    return x, y, z

def comoving_dist2comoving_vol(dist_min, dist_max, zmin=1e-6, zmax=0.4):
    # get the redshift at the comoving distance
    z_min = z_at_value(cosmo.comoving_distance, dist_min*u.Mpc, zmin=zmin, zmax=zmax, method='Bounded')
    z_max = z_at_value(cosmo.comoving_distance, dist_max*u.Mpc, zmin=zmin, zmax=zmax, method='Bounded')
    
    # calculate the volume from the redshift
    volume=(cosmo.comoving_volume(z_min)-cosmo.comoving_volume(z_max))
    return volume

def get_mean_std_bins(masses, sSFR, star_forming, min_lim=8, max_lim=12, step=0.2):
    bins_m = np.arange(min_lim, max_lim, step)
    M_bins=(bins_m[1:]+bins_m[:-1])*0.5

    # sSFR in mass bins
    mean_val, std_val = np.ones((len(bins_m[:-1]))), np.ones((len(bins_m[:-1])))
    for i, (ll, ul) in enumerate(zip(bins_m[:-1], bins_m[1:])):
        select= (masses[star_forming] > ll) & (masses[star_forming] <= ul)
        mean_val[i] = np.mean(sSFR[star_forming][select])
        std_val[i] =  np.std(sSFR[star_forming][select])
    return mean_val, std_val, M_bins

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

def normal_vector_spherical_surface(theta, phi):
    return [-np.cos(theta)*np.cos(phi), -np.sin(theta)*np.cos(phi), np.sin(phi)]


def comoving2redshift(d_c):
    """Gives you the redshift given the comoving distance of the object 

    d_c :: float 
        comoving distance in comoving kpc 
    """
    # first train the interpolation, get the requied x, y for this
    redshift_train = sigmoidspace(0, 0.3, 1000)
    d_c_train = cosmo.comoving_distance(redshift_train).value*1e3
    spl = CubicSpline(d_c_train, redshift_train)

    # get the redshift given the comoving distance as input
    redshift = spl(d_c)
    return redshift

def sigmoidspace(low,high,n,shape=5):
    raw = np.tanh(np.linspace(-shape,shape,n))
    return (raw-raw[0])/(raw[-1]-raw[0])*(high-low)+low

def mass2radius(mass, cosmo, z, delta = 200, ref_type = 'c'):
    """
    mass :: 
        in units of log_10()
    cosmo ::
        the type of cosmology used
    delta :: 
        the overdensity within which the quantity is measured
    ref_type ::
        the reference type, either 'c' for critical or 'm' or 'b' for mean/backgroup
    """
    if ref_type == 'c':
        density = cosmo.critical_density(z).to(u.Msun/u.kpc**3)
        radius = np.cbrt(((10**mass*u.Msun)/density)*(3/4*np.pi*delta))
        return radius
    else:
        raise ValueError(f"`ref_type` not recognized. {ref_type=}")
    
def rotation_matrix(coords):
    """Function provides the rotation matrix to center the y-z plane particles along the x-axis 

    Notes
    -----
        - Matrix obtained with scipy.spatial.transform.Rotation
        - see Notebook 04_applying_rotations_to_lightcone.ipynb for more details
        - validation of the matrix done in notebook 02_get_lightcone_geometry.ipynb
    """
    rot_mat = np.array([[ 9.95253462e-01, -8.85106062e-02, -4.04526723e-02],
       [ 8.85831154e-02,  9.96068789e-01,  2.16840434e-19],
       [ 4.02936443e-02, -3.58342374e-03,  9.99181456e-01]])
    coord_rot =  coords @ rot_mat

    coord_rot = coord_rot.astype(np.double)
    return coord_rot