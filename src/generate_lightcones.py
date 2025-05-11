"""
generate_lightcones.py contains the class to generate lightcones for gas particles, stellar particles, galaxies, and halos

Author: Soumya Shreeram
Date created: 16th April 2023
"""
import numpy as np
from astropy.table import QTable

import os
import logging 
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import seaborn as sns

# local imports
from tools import plotting as pt
from util import *
from remap_IllustrisTNG import Lightcone_IllustrisTNG
from get_geometry import Geometry 

class Generate_lightcones(Lightcone_IllustrisTNG):
    """Representation of generating lightcones for different TNG particle types
    
    Notes
    -----
    Refer to class `Lightcone_IllustrisTNG` for futher details
    """	
    def __init__(self, particle_type: str = 'Subhalo', observer_loc: int = 0, do_remapping: bool = False,
                 sim_data_dir: str = None, base_working_dir: str = None, out_data_dir: str = None,
                 clobber: bool = False, snap_nr: str = '98',
                 Lx_in=4.1231, Ly_in=0.7276, Lz_in=0.3333,
                 chunck_file: str = None, bin_no = None) -> None: 
        """
        Parameters
        ----------
        do_remapping :: bool 
        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)

        if particle_type is None:
            self.logger.info(f"{particle_type=}; options are `gas`, `dm`, `SubHalo`, `group`, `stars`")
            particle_type = 'gas'
            self.logger.info(f"Default set as {particle_type=}")
        
        super().__init__(particle_type=particle_type, snap_nr=snap_nr, chunck_file = chunck_file,
                         sim_data_dir = sim_data_dir, base_working_dir = base_working_dir, out_data_dir = out_data_dir)
        self.get_filenames(observer_loc = observer_loc, Lx_in=Lx_in, Ly_in=Ly_in, Lz_in=Lz_in, bin_no=bin_no)

        # set directory to the original remapping
        if self.particle_type.lower() in ['gas', 'gas_particles'] and self.chunck_file_number is not None:
            original_remap_coords = f"{self.snapshot_dir}/remap_Lx{Lx_in:.1f}_Ly{Ly_in:.1f}_Lz{Lz_in:.1f}/obs_0/coords_{self.chunck_file_number}.npy"            
        else:
            original_remap_coords = f"{self.snapshot_dir}/remap_Lx{Lx_in:.1f}_Ly{Ly_in:.1f}_Lz{Lz_in:.1f}/obs_0/coords.npy"

        # if the coordinates are not remapped 
        if not os.path.isfile(self.remapped_coords) or clobber:
            do_remapping = True
        
        self.original_remap_coords = original_remap_coords
        self.observer_loc = observer_loc
        self.do_remapping = do_remapping
        

    def produce_shells4observer(self, clobber = False) -> None:
        """Function to produce all possible orientations of the lightcones
        """
        # load the file that has the verticies and get the opening angles
        verticies = form_verticies(self.Lx, self.Ly, self.Lz)
        solver = Geometry(verticies=verticies, sphere_radius=self.Lx)
        
        if self.do_remapping and self.observer_loc == 0:
            self.do_boxremap(Lx_in=4.1231, Ly_in=0.7276, Lz_in=0.3333)

        if (not os.path.isfile(self.remapped_coords)) or clobber: 
            self.logger.info(f"{verticies[self.observer_loc]}")
            
            # load remapped coordinates and shift origin based on observer location 
            remapped_pos = np.load(self.original_remap_coords, allow_pickle=True) - verticies[self.observer_loc]
            np.save(self.remapped_coords, remapped_pos, allow_pickle=True)

        # convert coordinates to spherical
        self.convert_coords2spherical(clobber = clobber)

        # add signs based of observer location
        angular_cuts = solver.anglular_cuts(observer_loc=self.observer_loc)
        self.get_shells(angular_cuts[0][1].value, angular_cuts[1][1].value, theta_min=angular_cuts[0][0].value, phi_min=angular_cuts[1][0].value, clobber=clobber)
        return 
    
def plot_different_observer_lightcones(num_observers=8, clobber=False, single_obs=[False, 0], give_outputs=False):
    """Function to plot the different observer lightcones (for all 8 observers)

    Parameters
    ----------
    num_observers
    clobber
    single_obs
    give_outputs 
    """
    snapshot_list = np.arange(78, 100).astype(str)
    
    if single_obs[0]:
        observers_arr = [single_obs[1]]
    else:
        observers_arr = np.arange(num_observers)
    
    for observer_loc in observers_arr:    
        shell_id_arr = []
        radial_coord = []
        cart_coord = []
        for snapshot in snapshot_list:
            obs = Generate_lightcones(observer_loc = observer_loc, snap_nr=snapshot)
            if clobber:
                obs.produce_shells4observer(clobber=clobber)
            
            # save filenames
            shell_id_arr.append(obs.shell_ids)
            radial_coord.append(obs.spherical_remapped_coords)
            cart_coord.append(obs.remapped_coords)

        fig = plt.figure(figsize=(16, 25)) 
        ax = plt.axes(projection='3d')
        c_list = sns.color_palette("flare", len(shell_id_arr)).as_hex()
        for i, (id_names, coord_names) in enumerate(zip(shell_id_arr, cart_coord)):
            
            ids = np.load(id_names, allow_pickle=True)
            
            coord = np.load(coord_names, allow_pickle=True)
            
            
            x, y, z = coord[ids, 0][::100], coord[ids, 1][::100], coord[ids, 2][::100]
            ax.scatter3D(x, y, z, c=c_list[i], s=1e-1)
        pt.set_labels(ax, f'X [{obs.boxsize:.2f}]', 'Y', labelpad=[100, 20, 20], title=f'Observer {observer_loc}')
        
        # plot the edges estimated by simply getting the min/max in (x, y, z)
        remapped_coords = np.load(obs.remapped_coords, allow_pickle=True)
        edges_re = get_edges(remapped_coords)

        ax.scatter3D(edges_re[:, 0], edges_re[:, 1], edges_re[:, 2], marker='s', c='grey', s=15)
        ax.set_zlabel('Z')
        ax.set_box_aspect((obs.Lx, obs.Ly, obs.Lz))

    if give_outputs:
        output = shell_id_arr, radial_coord
    else:
        output = fig
    return output

def classify_centrals_sat_subhalos(snapshot, obs=0, column_names=['if_central', "mass_200c"],
                                   clobber=True,
                                   sim_data_dir: str = None, out_data_dir: str = None, base_working_dir: str = None):
        """Function to classify the galaxies into central and satellite
        """
        obs_of_subhalos = Generate_lightcones(observer_loc = obs, snap_nr=snapshot, particle_type = 'Subhalo',
                                              sim_data_dir = sim_data_dir,
                                              out_data_dir = out_data_dir, base_working_dir=base_working_dir)
        shell_ids = np.load(obs_of_subhalos.shell_ids, allow_pickle=True)

        # get the most massive (and resolved) subfind halos from the FoF halo catalog  
        obs_of_groups = Generate_lightcones(observer_loc = obs, snap_nr=snapshot, particle_type = 'group',
                                            sim_data_dir = sim_data_dir,
                                            out_data_dir = out_data_dir, base_working_dir=base_working_dir)
        t_group = QTable.read(obs_of_groups.lightcone_shell_properties, format='fits')
        groups = t_group['most_massive_subhalo'][t_group['most_massive_subhalo'] != -1]
        
        # set centrals as true
        centrals = np.isin(shell_ids, groups)
        
        # get the halo masses of these centrals 
        mass_200c = np.zeros(len(shell_ids))
        groups_with_centrals = np.isin(t_group['most_massive_subhalo'], shell_ids)
        mass_200c[np.where(centrals)] = t_group[np.where(groups_with_centrals)]["mass_200c"]
        
        # save theses values in columns 
        t_subhalo = QTable.read(obs_of_subhalos.lightcone_shell_properties, format='fits')
        if column_names not in t_subhalo.colnames:
            t_subhalo.add_columns([centrals, mass_200c], names=column_names)
            t_subhalo.write(obs_of_subhalos.lightcone_shell_properties, overwrite=clobber)
            logger.info(f"{column_names=} added to the table? {os.path.isfile(obs_of_subhalos.lightcone_shell_properties)}")
        return 
    
def merge_lightcone(clobber=False):
    """Function to create a joint lightcone file accross the lightcone
    """
    snapshot_list = np.flip(np.arange(78, 98).astype(str))
    t_all = []
    for i, snap_nr_j in enumerate(snapshot_list):
        logger.info(f"{snap_nr_j=}")
        # get the filenames    
        lc_j = Lightcone_IllustrisTNG(particle_type = 'Subhalo', snap_nr=snap_nr_j)
        lc_j.get_filenames()

        # check if file already exists
        if os.path.isfile(lc_j.lightcone_dir) and clobber and (i==0):
            os.remove(lc_j.lightcone_dir)
        
        # if creating the file for the first time
        test = QTable.read(lc_j.lightcone_dir, format='fits')
        t_j = QTable.read(lc_j.lightcone_shell_properties, format='fits')
        t_all.append(t_j)
    
    t_new = vstack(t_all, join_type='exact', metadata_conflicts='silent')
    t_new.write(lc_j.lightcone_dir, overwrite=True)