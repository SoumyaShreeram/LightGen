"""
.py contains the class MakeCuboid and Geometry, for obtaining the properties for getting intersections between a cuboid and spehere

Author: Soumya Shreeram
Date created: 2nd Mar 2023
"""
import numpy as np
import os
import logging 
from astropy import units as u

import util 

class MakeCuboid:
    """Representation of a cuboid
    :               (2)----------------(6)
    :              / |                / |
    :           (4)-----------------(7) |                     
    :            |   |               |  |
    :            |  (1)--------------|-(5)
    :            |  /                | /
    "            (3)----------------(8)
    """
    def __init__(self, verticies: np.array = None, filename_with_verticies: str = None, save_data_dir: str = None) -> None:
        """Initialized with the Pyramid object

        verticies :: shape(5, 3)
            the verticies defining the pyramid
        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)
        
        if verticies is None and filename_with_verticies is None:
            raise AttributeError("`verticies` is not provided. Must be an array of shape (5, 3). Else pass `filename_with_verticies` in .npy format.")
        
        if verticies is None and filename_with_verticies is not None:
            verticies = np.load(filename_with_verticies, allow_pickle=True)   
            self.logger.info(f"Read vertices from {filename_with_verticies=}")
            if verticies.shape == 0:
                raise ValueError(f"Emply file: {filename_with_verticies=}")
            self.verticies_file = filename_with_verticies

        if save_data_dir is None:
            save_data_dir = '/ptmp/soumyashreeram/cgmsim/IllustrisTNG'

        if verticies is not None and filename_with_verticies is None:
            print(os.path.isdir)
            np.save(f"{save_data_dir}/observer_locations.npy", verticies, allow_pickle=True)
            self.verticies_file = f"{save_data_dir}/observer_locations.npy"

        self.save_data_dir = save_data_dir

    def get_faces(self) -> dict:
        """Function to get the faces of the cuboid
        """
        verticies = np.load(self.verticies_file, allow_pickle=True)
        
        p1, p2, p3, p4 = verticies[:4]
        p5, p6, p7, p8 = verticies[4:]
  
        front_face = verticies[:4]    
        back_face = verticies[4:]
        
        bottom_face = np.vstack((p1, p3, p8, p5))
        top_face = np.vstack((p2, p4, p7, p6))

        front_side_face = np.vstack((p4, p3, p8, p7))
        back_side_face = np.vstack((p2, p1, p5, p6))

        faces_dict = {
            'front_face': front_face,
            'back_face': back_face,
            'bottom_face' : bottom_face,
            'top_face': top_face,
            'front_side_face' : front_side_face,
            'back_side_face'  : back_side_face,
        }
        return faces_dict

    def cuboid_edges(self) -> dict:  
        """Function to define the edges of the cuboid
        Note: top edge first, bottom second
        """
        verticies = np.load(self.verticies_file, allow_pickle=True)
        
        p1, p2, p3, p4 = verticies[:4]
        p5, p6, p7, p8 = verticies[4:]

        front_edges = np.vstack((p1, p8, p4, p7))
        back_edges = np.vstack((p2, p6, p1, p5))
        left_edges = np.vstack((p4, p2, p1, p3))
        right_edges = np.vstack((p7, p6, p8, p5))

        edges_dict = {
            'front_edges': front_edges,
            'back_edges': back_edges,
            'left_edges' :left_edges,
            'right_edges': right_edges
        }
        return edges_dict
        
class Geometry(MakeCuboid):
    """Representation of a Pyramid and it's properties
    """
    def __init__(self, verticies: np.array = None, filename_with_verticies: str = None, save_data_dir: str = None, sphere_radius: float = 4) -> None:
        """
        Parameters
        ----------
        sphere_radius :: float (default 4)
            the radius of the desired light cone
        """
        super().__init__(verticies, filename_with_verticies, save_data_dir)

        self.radius = sphere_radius

    def circle_line_intersection(self, r: float = None, center = (0, 0), line_param=(0, 0)):
        """Function to get the intersections between a circle and a line

        Notes: 
        -----
        (1) Equation of a line: y = mx + c 
        (2) Equation of a circle: (x-h)^2 + (y - k)^2 = r^2
        By subs for y using (1) in (2), we obtain a quadratic eqn whose roots we can solve for
        """
        if r is None:
            r = self.radius
        h, k = center
        m, c = line_param

        # obtain a quad eqn of the form: A*x**2 + B*x + C
        A = 1 + m**2
        B = 2*m*c - 2*h - 2*k*m 
        C = h**2 + k**2 + c**2 - r**2 - 2*k*c
        roots = np.roots([A, B, C])
        return roots


    def get_intersections(self, Lx=4.1231, Ly=0.7276, Lz=0.3333, verbrose=0):
        """Function to get the intersections between a sphere and a cuboid
        Representation of a cuboid 
    :               (2)--------(δ)-----(6)
    :              / |                / |
    :           (4)-------------(y)-(7) |                     
    :            |   |               |  |
    :            |  (1)----------(β)-|-(5)
    :            |  /                | /
    :            (3)----------------(a)
        """
        x0, y0, z0 = [0, 0, 0]
        alpha = Lx 

        # 1) Back face, botton edge intersection (X- Y plane with Z=0)
        beta = self.circle_line_intersection(center=(x0, y0), line_param=(x0, Ly))[0]
        

        # 2) Front face, top edge intersection (X-Z plane with Y=0)
        gamma = self.circle_line_intersection(center=(x0, y0), line_param=(x0, Lz))[0]
        

        # 3) Back face, top edge intersection (X-Y plane with Z=Lz and radius= $\delta$)
        delta = self.circle_line_intersection(r = beta, center=(0, Lz), line_param=(0, Ly))[0]
         
        # get the angles
        theta = util.get_angle([beta, Ly], [alpha, 0])*u.deg
        phi = util.get_angle([gamma, Lz], [alpha, 0])*u.deg
        ang1 = util.get_angle([beta, 0], [delta, Lz])*u.deg
        ang2 = util.get_angle([gamma, 0], [delta, Ly])*u.deg

        theta, phi = 180*u.deg-theta, 180*u.deg-phi
        self.logger.info(f"{theta=:.2f}, {phi=:.2f}, {ang1=:.2f}, {ang2=:.2f}")
        dict_intersections = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta
        }
        return theta, phi, ang1, ang2, dict_intersections
    
    def anglular_cuts(self, observer_loc = None):
        """Based on the observer location, decides the azimuthal cuts 

        Parameters
        ----------
        theta: float
            the 
        """
        theta, phi, _, _, _ = self.get_intersections()
        
        dict_cuts = {
            'obs_0':[ [0*u.deg, theta], [0*u.deg, phi] ],
            'obs_1':[ [0*u.deg, theta], [-phi, 0*u.deg] ],
            'obs_2':[ [-theta, 0*u.deg], [0*u.deg, phi] ],
            'obs_3':[ [-theta, 0*u.deg], [-phi, 0*u.deg] ],
            'obs_4':[ [-theta, 0*u.deg], [0*u.deg, phi] ],
            'obs_5':[ [-theta, 0*u.deg], [-phi, 0*u.deg] ],
            'obs_6': [[0*u.deg, theta], [0*u.deg, phi] ], 
            'obs_7': [[0*u.deg, theta], [-phi, 0*u.deg] ], 
        }
        if observer_loc is None:
            observer_loc = 0
        return dict_cuts[f'obs_{observer_loc}']