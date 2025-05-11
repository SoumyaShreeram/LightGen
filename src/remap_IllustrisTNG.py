"""
remap_IllustriTNG.py contains the class to generate lightcones from the hydro-dynamical simulation outputs of IllustrisTNG

Author: Soumya Shreeram
Date created: 15th Feb 2023
"""
import numpy as np
import pandas as pd
import os
import logging 
import h5py

# all astropy imports
from astropy.cosmology import FlatLambdaCDM
# input cosmology from the IllustrisTNG simulations:
cosmo = FlatLambdaCDM(H0=0.6774*100, Om0=0.3089, Tcmb0=2.7255, Ob0=0.0486, Neff=3.046)
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.table import QTable

import matplotlib as mpl
import matplotlib.pyplot as plt

# local importsls
from tools import illustris_python as il
from tools import plotting as pt
from tools import remap as re
import util 


class Lightcone_IllustrisTNG:
    """Representationo of generating a lightcone from snapshots

    Notes:
    -----
    Data access required to the following directory on raven:
        /virgotng/universe/IllustrisTNG/TNG300-1/output/

    For running scripts on raven, data is copied at:
        /ptmp/soumyashreeram/TNGproducts

    Refer to the following webpage for details of the chosen simulation:
        www.tng-project.org/data/downloads/TNG300-1/

    Methods:
    -------
    self.snapshot2redshifts() ::
        converts the snapshot numbers to redshift
    self.get_filename() ::
        Gets the filename of the simulation snapshots for a given particle type
    self.get_particles_coord() ::
        Gets and saves the X, Y, Z postions of the snapshot box
    self.get_transformation_matrix() ::
        Obtains the transformation matrix for remapping the original snapshot
    self.do_boxremap() ::
        Performs the remapping using the boxremap package
    self.do_boxremap_fast_test() ::
        Performs a remapping test with fewer points using the boxremap package 

    """
    def __init__(self, sim_data_dir: str = None, base_working_dir: str = None,
                 out_data_dir: str = None, snap_nr: str = None,
                 save_plots_dir: str = None,
                 particle_type: str = 'gas', boxsize = 205000, verbrose=0,
                 chunck_file: str = None) -> None:
        """Initialized with the lightcone object

        Parameters:
        ----------
        sim_data_dir :: str
            the directory where the TNG snapshots are stored
        base_working_dir :: str
            the directory where the code/package is developed
        out_data_dir :: str
            the directory where the data products generated are outputted/written
        snap_nr :: str
            the snap shot number
        save_plots_dir :: str
            the directory where all the generated plots are saved
        particle_type :: str
            the particle type for which the operations are performed (default set to `gas`)
        boxsize :: float
            the boxsize of the simulation used. In units of ckpc/h
        """
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(type(self).__name__)

        if sim_data_dir is None:
            raise ValueError(f"Please pass the directory where you store the N-body simulation file to `sim_data_dir`. Currently {sim_data_dir=}")
        
        if base_working_dir is None:
            base_working_dir = os.getcwd()
            self.logger.info(f"Your `base_working_dir` is set to {base_working_dir}. Pass another path if needed.")

        if out_data_dir is None:
            out_data_dir = f"{os.getcwd()}/outputs"
            if not os.path.isdir(out_data_dir):
                os.mkdir(out_data_dir)
            self.logger.info(f"Your `out_data_dir` is set to {out_data_dir}. Pass another path to keyword if needed.")
        self.out_data_dir = out_data_dir

        if snap_nr is None:
            if verbrose !=0:
                self.logger.info(f"{snap_nr=}; using default value =`99`")
            snap_nr = "99"

        if save_plots_dir is None:
            save_plots_dir = f'{out_data_dir}/plots'
            if not os.path.isdir(save_plots_dir):
                os.mkdir(save_plots_dir)

        redshift = self.snapshot2redshifts(snap_nr=snap_nr)
        if verbrose != 0:
            self.logger.info(f"{redshift=}")
        snapshot_dir = f"{out_data_dir}/{particle_type}_SN{snap_nr}"
        if not os.path.isdir(snapshot_dir):
            os.mkdir(snapshot_dir)

        if particle_type is None:
            self.logger.info(f"{particle_type=}; options are `gas`, `dm`, `SubHalo`, `Group`, `bh`")
            particle_type = 'gas'
            self.logger.info(f"Default set as {particle_type=}")

        h = (cosmo.H0/100).value # in units of (km/s)/Mpc
        boxsize_Mpc = boxsize/h * 1e-3 * u.Mpc # in units of cMpc

        if str(particle_type).lower() in ['gas', 'star', 'stars'] and chunck_file is not None:
            start = chunck_file.find(f'snap_0{snap_nr}.') + 9
            end = chunck_file.find('.hdf5', start)
            chunck_file_number = chunck_file[start:end]
        else:
            chunck_file_number = None

        self.h = h
        self.sim_data_dir = sim_data_dir
        self.base_working_dir = base_working_dir
        self.snap_nr = snap_nr
        self.redshift = redshift
        self.save_plots_dir = save_plots_dir
        self.particle_type = particle_type
        self.snapshot_dir = snapshot_dir
        self.boxsize = boxsize_Mpc
        self.verbrose = verbrose
        self.chunck_file_number = chunck_file_number
        self.chunck_file = chunck_file

    def snapshot2redshifts(self, snap_nr="99", clobber=False) -> float:
        """This function outputs the redshift for the input snapshot
        Parameters
        ----------
        snap_nr :: str (default == "99")
            the snapshot number
        Return
        -----
        redshift of the corresponding snapshot
        """
        if not isinstance(snap_nr, str):
            raise Exception(f'{snap_nr} must be a str')
        snapshot_z_dict = {
            "78": 0.30,
            "79": 0.27,
            "80": 0.26,
            "81": 0.24,
            "82": 0.23,
            "83": 0.21,
            "84": 0.20, 
            "85": 0.18,
            "86": 0.17,
            "87": 0.15,
            "88": 0.14,
            "89": 0.13,
            "90": 0.11,
            "91": 0.10,
            "92": 0.08,
            "93": 0.07,
            "94": 0.06,
            "95": 0.05,
            "96": 0.03,
            "97": 0.02,
            "98": 0.01,
            "99": 0.00
            }
        z = [0.30,0.27,0.26,0.24,0.23,0.21,0.20,0.18,0.17,0.15,0.14,0.13,0.11,0.10,0.08,0.07,0.06,0.05,0.03,0.02,0.01,0.00]
        arr = np.array([np.arange(78, 100), np.array(z)]).T
        
        snapshot_info_file = f"{self.out_data_dir}/snapshot_info.csv"
        if clobber and os.path.isfile(snapshot_info_file):
            os.remove(snapshot_info_file)

        if not os.path.isfile(snapshot_info_file):
            df = pd.DataFrame(arr, columns=['snapshot','redshift'])
            df.to_csv(snapshot_info_file, header=True, sep=' ')
        self.snapshot_info_file = snapshot_info_file

        return snapshot_z_dict[snap_nr]

    def get_filenames(self, Lx_in=4.1231, Ly_in=0.7276, Lz_in=0.3333, observer_loc: int = 0, bin_no: int = None) -> None:
        """Function to get the filenames of all the products that will be/ are generated in this class

        Parameters
        ----------
        Lx_in, Ly_in, Lz_in :: (float, float, float)
            the (normalized) dimensions of the box to which the original box is remapped
        observer_loc :: int
            the location of the observer
        chunck_file_number ::  int
            the file number within a snapshot/group directory
        return_filenames :: bool
            decides wether to return the snapshot/group directory name
        bin_no :: int
            the bin no of the subshell (different sampling of particles form chunks to subshells)
            can range from 1-12
        Returns
        -------
        self.remapped_coord_dir :: str
            the directory where the remapped coordinates are saved
        self.observer_loc :: int
            the location of the observer, can run from  0-7 
        self.Lx, self.Ly, self.Lz :: (float, float, float)
            the (normalized) dimensions of the box to which the original box is remapped
        self.observer_orient_dir :: str
            the directory for the given observer location 
        self.remapped_coords :: str
            the file where the coord are remapped
        self.spherical_remapped_coords :: str 
            the file where the remapped coord are converted into spherical coords
        self.shell_ids :: str
            the file with the indicies of the relavant particles/Subhalos in the shell (defined by snapshot redshift)
        self.lightcone_shell_properties :: 
            the properties of the particles in the given shell
        """
        # generated in self.do_boxremap()
        self.original_positions = f"{self.snapshot_dir}/original_positions.npy"
        remapped_coord_dir =  f"{self.snapshot_dir}/remap_Lx{Lx_in:.1f}_Ly{Ly_in:.1f}_Lz{Lz_in:.1f}"
        if not os.path.isdir(remapped_coord_dir):
            os.mkdir(remapped_coord_dir)
        
        observer_orient_dir = f"{remapped_coord_dir}/obs_{observer_loc}"
        if not os.path.isdir(observer_orient_dir):
            os.mkdir(observer_orient_dir)      

        # ===========================================
        # coordinates/results are in chunck formats for gas particles to reduce computational time
        if self.particle_type.lower() in ['gas', 'gas_particles', 'star', 'stars'] and self.chunck_file_number is None:
            raise ValueError("Please pass snapshots in subfiles")

        # for specified subfile case of gas particles
        elif self.particle_type.lower() in ['gas', 'gas_particles', 'star', 'stars'] and self.chunck_file_number is not None:
            remapped_coords = f"{observer_orient_dir}/coords_{self.chunck_file_number}.npy"

            # generated in self.convert_coords2spherical() and self.get_shells()
            spherical_remapped_coords = f"{observer_orient_dir}/spherical_coords_{self.chunck_file_number}.npy"
            shell_ids = f"{observer_orient_dir}/shell_ids_{self.chunck_file_number}.npy"

            # for reorganizing the particles within the lc
            self.subshell_bins = f"{observer_orient_dir}/subshell_bins.npy"
            self.subshell_dir = f"{observer_orient_dir}/subshells"
            if not os.path.isdir(self.subshell_dir):
                os.mkdir(self.subshell_dir)
            self.subshell_ids = f"{self.subshell_dir}/subshell_ids_{self.chunck_file_number}.npy"
            self.stored_subshell_quantities = f"{self.subshell_dir}/Quantities_{self.chunck_file_number}.hdf5"

            if bin_no is not None:
                self.chunkfiles_in_bin =f"{self.subshell_dir}/chunk_nos_in_lightcone_subshell_bin_{bin_no}.npy"
                self.bin_no = bin_no
            else:
                self.chunkfiles_in_bin =f"{observer_orient_dir}/chunk_nos_in_lightcone.npy"

            # names of the particles in subshells
            if os.path.isfile(self.subshell_bins):
                bins = np.load(self.subshell_bins, allow_pickle=True)
                self.bin_names = [f"{self.subshell_dir}/bin_no_{i}_{bins[i-1]:.2f}_{bins[i]:.2f}_cMpc.npy" for i in np.arange(1, len(bins), dtype=int)]
                if bin_no is not None: 
                    if bin_no > len(bins): # number if subshells varies with the snashot shell itself
                        raise AttributeError(f"{bin_no=} is larger than the number of bins (lightcone shell in snapshot {self.snap_nr} has {len(bins)} bins)")
                    self.stored_bin_subshell_quantities = f"{self.subshell_dir}/Quantities_bin_{self.bin_no}_{bins[self.bin_no-1]:.2f}_{bins[self.bin_no]:.2f}_cMpc.h5"
                    
            # generated after running self.get_lightcone_properties()
            lc_properties_file = f"{observer_orient_dir}/lightcone_shell_{self.chunck_file_number}.fits"

            # generated for stars only
            self.age_info_file = f"{observer_orient_dir}/birth_scale_factor_{self.chunck_file_number}.npy"
        else:
            remapped_coords = f"{observer_orient_dir}/coords.npy"
            spherical_remapped_coords = f"{observer_orient_dir}/spherical_coords.npy"
            shell_ids = f"{observer_orient_dir}/shell_ids.npy"
            lc_properties_file = f"{observer_orient_dir}/lightcone_shell.fits"
        # ===========================================

        # saved file paths
        self.remapped_coord_dir = remapped_coord_dir
        self.observer_loc = observer_loc
        self.Lx, self.Ly, self.Lz = Lx_in, Ly_in, Lz_in
        self.observer_orient_dir = observer_orient_dir
        
        # products required for lightcone generation
        self.remapped_coords = remapped_coords
        self.spherical_remapped_coords = spherical_remapped_coords
        self.shell_ids = shell_ids
        self.lightcone_shell_properties = lc_properties_file

        # directory to save lightcones for different particle types
        self.lightcone_dir = f"{self.out_data_dir}/Lightcones/{self.particle_type}_remap_Lx{Lx_in:.1f}_Ly{Ly_in:.1f}_Lz{Lz_in:.1f}_obs{self.observer_loc}.fits"
        return 

    def get_particles_coord(self, plot_particles: bool = False, pos_path: str = None, clobber: bool = False):
        """Function loads the coordinates of particles/cells of a given parttype for every sub-file within a snapshot
        Parameters
        ----------
        plot_gas_particles :: bool
            decides wether or not to plot the gas particle coordinates
        pos_path :: str
            the path where the particle positions (x, y, z) of the original box (Lx, Ly, Lz=1) are stored
        """
        if pos_path is None:
            pos_path = f"{self.snapshot_dir}/original_positions.npy"
        elif self.particle_type.lower() in ['gas', 'gas_particles', 'star', 'stars'] and self.chunck_file_number is not None:
            pos_path =  f"{self.snapshot_dir}/original_positions_{self.chunck_file_number}.npy"

        self.original_positions_path = pos_path
        if not os.path.isfile(pos_path) or clobber:
            if str(self.particle_type).lower() in ['dm', 'bh', 'blackholes', 'bhs']:
                pos = il.snapshot.loadSubset(self.sim_data_dir, self.snap_nr, self.particle_type, fields=['Coordinates'])
                
            if str(self.particle_type).lower() in ['gas', 'gas_particles', 'star', 'stars']:
                partTypeNum = il.util.partTypeNum(self.particle_type)
                with h5py.File(self.chunck_file, 'r') as f:
                    pos = f[f'PartType{partTypeNum}/Coordinates'][:]
            
            if self.particle_type == 'Subhalo':
                pos = il.groupcat.loadSubhalos(self.sim_data_dir, self.snap_nr, fields=['SubhaloPos'])
            
            if str(self.particle_type).lower() in ['group', 'halo', 'fof halo']:
                pos = il.groupcat.loadHalos(self.sim_data_dir, self.snap_nr, fields=['GroupPos'])

            # print out info    
            if self.verbrose != 0:
                self.logger.info(f"{pos.shape} {self.particle_type} particles for {self.snap_nr=}")
                
            pos = pos/(self.h*1e3*self.boxsize.value)
            if self.chunck_file_number is None:
                np.save(pos_path, pos, allow_pickle=True)
        else: 
            pos = np.load(pos_path, allow_pickle=True)

        if plot_particles:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.hist2d(pos[::100,0], pos[::100,1], norm=mpl.colors.LogNorm(), bins=100)
            pt.set_labels(ax, 'x [ckpc/h]', 'y [ckpc/h]', title=f'Suhalo sub-snapshot SN{self.snap_nr}')
            fig.savefig(f"{self.save_plots_dir}/{self.particle_type}_particles_SN{self.snap_nr}.png")
        return pos

    def get_transformation_matrix(self, Lx_in, Ly_in, Lz_in):
        """Function to obatain the transformation matrix 

        Parameters
        ----------
        Lx_in, Ly_in, Lz_in ::  (float, float, float)
            the (normalized) dimensions of the box to which the original box is remapped
        """
        a = np.loadtxt(f'{self.base_working_dir}/tools/list7.txt', dtype=str)
        Lx, Ly, Lz = a[:, 0].astype(np.float64), a[:, 1].astype(np.float64), a[:, 2].astype(np.float64)
        common_slots = np.where((Lx == Lx_in) & (Ly == Ly_in) & (Lz == Lz_in))[0]

        if len(common_slots) == 0:
            raise Exception(f"Did not find a transformation matrix for {Lx_in=}, {Ly_in=}, {Lz_in=}")
        id = common_slots[0]

        u1 = (a[id, 3].astype(int), a[id, 4].astype(int), a[id, 5].astype(int))
        u2 = (a[id, 6].astype(int), a[id, 7].astype(int), a[id, 8].astype(int))
        u3 = (a[id, 9].astype(int), a[id, 10].astype(int), a[id, 11].astype(int))
        return u1, u2, u3

    def do_boxremap(self, Lx_in, Ly_in, Lz_in, plot_transformed_coords: bool = False, observer_loc: int = 0, clobber: bool = False, bins: int = 164) -> None:
        """Function for remapping the box to another desired size

        Parameters
        ----------
        Lx_in, Ly_in, Lz_in ::  (float, float, float)
            the (normalized) dimensions of the box to which the original box is remapped
        plot_transformed_coords :: bool (default True)
            decides wether or not to plot the remapped coordinates
        observer_loc :: int (default 0)
            the location where the observer is set at in the remapped box
        clobber :: bool (default False)
            decided wether or not to regenerate the files
        Notes:
        -----
        Check out file cgmsim/tools/list7.txt to obtain the transformation matrix

        E.g., to obtain a box size of Lx, Ly, Lz = (4.1231, 0.3430, 0.7071)
        The transformation matrix is (u1, u2, u3)
        """
        self.get_filenames()

        # do the remapping
        if not os.path.isfile(self.remapped_coords) or clobber:
            pos = self.get_particles_coord(clobber=clobber)
            self.logger.info("Got original positions")
            # get the transformation matrix
            u1, u2, u3 = self.get_transformation_matrix(Lx_in, Ly_in, Lz_in)

            # remap using the transformation matrix
            C = re.Cuboid(u1=u1, u2=u2, u3=u3)

            # obtain the transformed coordinated
            pos_t = np.zeros(np.shape(pos))
            X_t, Y_t, Z_t = np.zeros(np.shape(pos[:,0])), np.zeros(np.shape(pos[:,1])), np.zeros(np.shape(pos[:,2]))
            self.logger.info("transforming coordinates now ")
            for i, (x, y, z) in enumerate(zip(pos[:,0], pos[:,1], pos[:,2])):
                X_t[i], Y_t[i], Z_t[i] = C.Transform(x, y, z)
            pos_t[:,0], pos_t[:,1], pos_t[:,2] = X_t, Y_t, Z_t

            np.save(self.remapped_coords, pos_t, allow_pickle=True)
        else:
            pos_t = np.load(self.remapped_coords, allow_pickle=True)

        # check the remapping with a projected x-y plot
        if plot_transformed_coords:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.hist2d(pos_t[::100,0], pos_t[::100,1], norm=mpl.colors.LogNorm(), bins=bins)
            pt.set_labels(ax, 'x [ckpc/h]', 'y [ckpc/h]', title=f'Suhalo sub-snapshot SN{self.snap_nr}')
            fig.savefig(f"{self.save_plots_dir}/{self.particle_type}_particles_SN{self.snap_nr}_remapped_Lx{Lx_in:.1f}_Ly{Ly_in:.1f}_Lz{Lz_in:.1f}.png")

        # saved file paths
        self.observer_loc = observer_loc
        self.Lx, self.Ly, self.Lz = Lx_in, Ly_in, Lz_in
        return

    def get_lightcone_shell_boundaries(self, zmin=1e-7, zmax=0.45, clobber=False) -> None:
        """Function to get the shells boundaries of the lightcone fromo the snapshot boxes
        """
        # create and save the snapshot info used
        self.snapshot2redshifts(clobber=clobber)
        snapshot_info = pd.read_csv(self.snapshot_info_file, sep=' ', header=0)

        d_c = cosmo.comoving_distance(snapshot_info['redshift'].to_numpy()).value # unit is cMpc

        # midpoints i.e. points at which we switch between outputs of a snapshot
        mid_points =(d_c[1:]+d_c[:-1])/2
        d_c_max_boundaries = np.append(np.array([d_c[0]]), mid_points)
        d_c_min_boundaries = np.append(mid_points, d_c[-1])

        # get the redshifts of the boundaries and the volume of the shell
        redshift_min = z_at_value(cosmo.comoving_distance, d_c_min_boundaries[:-1]*u.Mpc, zmin=zmin, zmax=zmax, method='Bounded')
        redshift_max = z_at_value(cosmo.comoving_distance, d_c_max_boundaries[:-1]*u.Mpc, zmin=zmin, zmax=zmax, method='Bounded')
        volume=(cosmo.comoving_volume(redshift_max)-cosmo.comoving_volume(redshift_min)).value
        redshift_min, redshift_max, volume = np.append(redshift_min.value, 0), np.append(redshift_max.value, 0), np.append(volume, 0)

        d = {'snapshot': snapshot_info['snapshot'].to_numpy(),
             'redshift': snapshot_info['redshift'].to_numpy(),
             'd_c': d_c,
             'd_c_min': d_c_min_boundaries,
             'd_c_max': d_c_max_boundaries,
             'redshift_min': redshift_min,
             'redshift_max': redshift_max,
             'volMpc3': volume}

        df = pd.DataFrame(data=d)
        df.to_csv(self.snapshot_info_file, header=True, sep=' ')
        return

    def convert_coords2spherical(self, clobber: bool = False):
        """Function to convert the cartesian coordinated into spherical
        Saves the file with the (x, y, z) -> (r, altitude, azimuth)
        Corresponds to the coordinate system where the observer is at the origin
        """
        self.get_filenames()

        if not os.path.isfile(self.spherical_remapped_coords) or clobber:
            pos = np.load(f"{self.remapped_coords}", allow_pickle=True)
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2] # these are unitless (go from 0-1)

            spherical_pos = util.cartesian2polar(x, y, z)

            # save this info
            np.save(self.spherical_remapped_coords, spherical_pos,  allow_pickle=True)
        return

    def get_shells(self, theta_max, phi_max, theta_min=0, phi_min=0, clobber: bool = False)-> None:
        """Function to get the lightcone shells

        Parameters
        ----------
        theta_max :: float
            the max opening angle in the X-Y plane
        phi_max :: float
            the max opening angle in the X-Z plane
        snap_nr :: float
            the snapshot number of the IllustrisTNG box
        clobber :
            decides wether or nnot to rerum the function
        """
        self.get_filenames()
        if not os.path.isfile(self.shell_ids) or clobber:
            # load the snapshot info
            sph_remapped_coords = np.load(self.spherical_remapped_coords,  allow_pickle=True).T
            r, theta, phi = sph_remapped_coords[:, 0], sph_remapped_coords[:, 1], sph_remapped_coords[:, 2]

            # load the spherical coordinates
            f = pd.read_csv(self.snapshot_info_file, sep=' ')
            if 'd_c_min' not in list(f.columns):
                self.get_lightcone_shell_boundaries()
            get_snap_nr = np.where(f['snapshot'].to_numpy() == float(self.snap_nr))[0][0]

            r_min, r_max = f['d_c_min'].to_numpy()[get_snap_nr], f['d_c_max'].to_numpy()[get_snap_nr] # in Mpc

            # convert to unitless quantities
            r_min, r_max = r_min/self.boxsize.value, r_max/self.boxsize.value
            if self.verbrose != 0: 
                self.logger.info(f"{r_min=:.2f}, {r_max=:.2f} units")

            r_coords = (r > r_min) & (r <= r_max)
            angular_coords = (theta_min < np.rad2deg(theta)) & (np.rad2deg(theta) <= theta_max) & (phi_min < np.rad2deg(phi)) & (np.rad2deg(phi) <= phi_max)
            ids = np.where(r_coords & angular_coords)[0]
            self.logger.info(f"{self.snap_nr=}, selected={ids.shape} {self.particle_type}")

            # save this info
            np.save(self.shell_ids, ids,  allow_pickle=True)
        return 

    def get_lightcone_properties(self, zmin=1e-6, zmax=0.4, clobber=False) -> None:
        """Function to get the required properties of the particles based of validation needs
        This function redirects to the respective particle specific functions
        """
        if self.particle_type.lower() == 'subhalo':
            self.get_lightcone_properties_subhalo(zmin=zmin, zmax=zmax, clobber=clobber)

        if self.particle_type.lower() in ['group', 'halo', 'fof halo']:
            self.get_lightcone_properties_halo(zmin=zmin, zmax=zmax, clobber=clobber)

        if self.particle_type.lower() in ['agn', 'blackholes', 'bh', 'bhs']:
            self.get_lightcone_properties_bh(clobber=clobber)

    def get_lightcone_properties_bh(self, clobber=False) -> None:
        """Function to get the required properties of the black holes in the lightcone 
        """
        lc_properties_file = f"{self.observer_orient_dir}/lightcone_shell.fits"
        if clobber and os.path.isfile(lc_properties_file):
            os.remove(lc_properties_file)

        if not os.path.isfile(lc_properties_file) or clobber:
            self.logger.info(f" Generating fits table with required properties for {self.particle_type=} for {self.observer_loc=}") 

            # load the selected particles within the shells defined by the snapshot
            shell_ids = np.load(self.shell_ids)

            # get all useful quantities
            BH_HostHaloMass = il.snapshot.loadSubset(self.sim_data_dir, self.snap_nr, self.particle_type, fields=['BH_HostHaloMass'])[shell_ids]*(10**10*u.Msun/(self.h))
            BH_Mass = il.snapshot.loadSubset(self.sim_data_dir,self.snap_nr, self.particle_type, fields=['BH_Mass'])[shell_ids]*((10**10*u.Msun/(self.h)))
            BH_Mdot = il.snapshot.loadSubset(self.sim_data_dir,self.snap_nr,self.particle_type,  fields=['BH_Mdot'])[shell_ids]*((10**10*u.Msun/(self.h)) / (0.978*u.Gyr/(self.h)))
            CumEgyInjection_QM = il.snapshot.loadSubset(self.sim_data_dir,self.snap_nr, self.particle_type, fields=['BH_CumEgyInjection_QM'])[shell_ids]*((10**10*u.Msun/(self.h)) * (u.kpc/self.h)**2 / (0.978*u.Gyr)**2)
            CumEgyInjection_RM = il.snapshot.loadSubset(self.sim_data_dir,self.snap_nr, self.particle_type, fields=['BH_CumEgyInjection_RM'])[shell_ids]*((10**10*u.Msun/(self.h)) * (u.kpc/self.h)**2 / (0.978*u.Gyr)**2)
            BH_Progs = il.snapshot.loadSubset(self.sim_data_dir,self.snap_nr, self.particle_type, fields=['BH_Progs'])[shell_ids]

            # load the snapshot info
            sph_remapped_coords = np.load(self.spherical_remapped_coords,  allow_pickle=True).T
            d_c = sph_remapped_coords[shell_ids, 0]*self.boxsize

            pos = np.load(f"{self.remapped_coords}", allow_pickle=True)
            x, y, z = pos[shell_ids, 0]*self.boxsize, pos[shell_ids, 1]*self.boxsize, pos[shell_ids, 2]*self.boxsize

            # load the shell boundaries for meta data
            f = pd.read_csv(self.snapshot_info_file, sep=' ')
            get_snap_nr = np.where(f['snapshot'] == float(self.snap_nr))[0][0]
            r_min, r_max = f['d_c_min'].to_numpy()[get_snap_nr], f['d_c_max'].to_numpy()[get_snap_nr]

            t = QTable([BH_HostHaloMass, BH_Mass, BH_Mdot, CumEgyInjection_QM, CumEgyInjection_RM, BH_Progs, d_c, x, y, z],
            names=("BH_HostHaloMass", "BH_Mass", "BH_Mdot", "CumEgyInjection_QM", "CumEgyInjection_RM", "BH_Progs", "d_c", "x", "y", "z"),
            meta={'SnapnNo': f"{self.snap_nr}",
                  'shellMin': f"{r_min:.2f} Mpc",
                  'shellMax': f"{r_max:.2f} Mpc" })

            t.write(lc_properties_file, overwrite=True)
            self.logger.info(f" File generated? {os.path.isfile(lc_properties_file)}")

        self.lightcone_shell_properties = lc_properties_file
        return

    def get_lightcone_properties_halo(self, zmin=1e-6, zmax=0.4, clobber=False) -> None:
        """Function to get the required properties halos in the lightcone 

        Parameters
        ----------
        snap_nr :: str 
            the snapshot number 
        zmin, zmax :: float, float
            the min and max redshift of the lightcone
        clobber :: bool
            decides wether to rerun the code
        """
        lc_properties_file = f"{self.observer_orient_dir}/lightcone_shell.fits"
        if clobber and os.path.isfile(lc_properties_file):
            os.remove(lc_properties_file)

        if not os.path.isfile(lc_properties_file) or clobber:
            self.logger.info(f" Generating fits table with required properties for {self.observer_loc=}") 

            # load the selected particles within the shells defined by the snapshot
            shell_ids = np.load(self.shell_ids)

            # get all useful quantities
            most_massive_subhalo = il.groupcat.loadHalos(self.sim_data_dir, self.snap_nr, fields=['GroupFirstSub'])[shell_ids]
            mass_200c = il.groupcat.loadHalos(self.sim_data_dir,self.snap_nr, fields=['Group_M_Crit200'])[shell_ids]*(10**10*u.Msun/self.h)
            radius_200c = il.groupcat.loadHalos(self.sim_data_dir,self.snap_nr, fields=['Group_R_Crit200'])[shell_ids]*(u.kpc/self.h)
            mass_500c = il.groupcat.loadHalos(self.sim_data_dir,self.snap_nr, fields=['Group_M_Crit500'])[shell_ids]*(10**10*u.Msun/self.h)
            radius_500c = il.groupcat.loadHalos(self.sim_data_dir,self.snap_nr, fields=['Group_R_Crit500'])[shell_ids]*(u.kpc/self.h)

            # load the snapshot info
            sph_remapped_coords = np.load(self.spherical_remapped_coords,  allow_pickle=True).T
            d_c = sph_remapped_coords[shell_ids, 0]*self.boxsize

            pos = np.load(f"{self.remapped_coords}", allow_pickle=True)
            x, y, z = pos[shell_ids, 0]*self.boxsize, pos[shell_ids, 1]*self.boxsize, pos[shell_ids, 2]*self.boxsize

            # calculate the redshift of each particle
            redshift = z_at_value(cosmo.comoving_distance, d_c, zmin=zmin, zmax=zmax, method='Bounded')
            redshift = np.array(redshift, dtype=float)
            
            # load the shell boundaries for meta data
            f = pd.read_csv(self.snapshot_info_file, sep=' ')
            get_snap_nr = np.where(f['snapshot'] == float(self.snap_nr))[0][0]
            r_min, r_max = f['d_c_min'].to_numpy()[get_snap_nr], f['d_c_max'].to_numpy()[get_snap_nr]

            t = QTable([most_massive_subhalo, mass_200c, mass_500c, d_c, x, y, z, redshift, radius_200c, radius_500c],
            names=("most_massive_subhalo", "mass_200c", "mass_500c", "d_c", "x", "y", "z", "redshift", "radius_200c", "radius_500c"),
            meta={'SnapnNo': f"{self.snap_nr}",
                  'shellMin': f"{r_min:.2f} Mpc",
                  'shellMax': f"{r_max:.2f} Mpc" })
            
            t.write(lc_properties_file, overwrite=True)
            self.logger.info(f" File generated? {os.path.isfile(lc_properties_file)}")
        
        self.lightcone_shell_properties = lc_properties_file
        return 
    
    def get_lightcone_properties_subhalo(self, zmin=1e-6, zmax=0.4, clobber=False) -> None:
        """Function to get the required properties subhalos in the lightcone 

        Parameters
        ----------
        snap_nr :: str 
            the snapshot number 
        zmin, zmax :: float, float
            the min and max redshift of the lightcone
        clobber :: bool
            decides wether to rerun the code
        """
        lc_properties_file = f"{self.observer_orient_dir}/lightcone_shell.fits"
        if clobber and os.path.isfile(lc_properties_file):
            os.remove(lc_properties_file)

        if not os.path.isfile(lc_properties_file) or clobber:
            self.logger.info(f" Generating fits table with required properties for {self.observer_loc=}") 

            # load the selected particles within the shells defined by the snapshot
            shell_ids = np.load(self.shell_ids)

            # get all useful quantities
            flag = il.groupcat.loadSubhalos(self.sim_data_dir, self.snap_nr, fields=['SubhaloFlag'])[shell_ids]
            tot_mass = il.groupcat.loadSubhalos(self.sim_data_dir,self.snap_nr, fields=['SubhaloMass'])[shell_ids]*(10**10*u.Msun/self.h)
            radius = 2*il.groupcat.loadSubhalos(self.sim_data_dir,self.snap_nr, fields=['SubhaloHalfmassRadType'])[:, 4][shell_ids]*(u.kpc/self.h)
            stellar_mass = il.groupcat.loadSubhalos(self.sim_data_dir,self.snap_nr, fields=['SubhaloMassInRadType'])[:, 4][shell_ids]*(10**10*u.Msun/self.h)
            sfr = il.groupcat.loadSubhalos(self.sim_data_dir,self.snap_nr, fields=['SubhaloSFRinRad'])[shell_ids]*(u.Msun/u.yr)

            # load the snapshot info
            sph_remapped_coords = np.load(self.spherical_remapped_coords,  allow_pickle=True).T
            d_c = sph_remapped_coords[shell_ids, 0]*self.boxsize # in units of cMpc

            pos = np.load(f"{self.remapped_coords}", allow_pickle=True)
            x, y, z = pos[shell_ids, 0]*self.boxsize, pos[shell_ids, 1]*self.boxsize, pos[shell_ids, 2]*self.boxsize

            # calculate the redshift of each particle
            self.logger.info(f"{d_c.value=}")
            redshift = util.comoving2redshift(d_c.value)

            # load the shell boundaries for meta data
            f = pd.read_csv(self.snapshot_info_file, sep=' ')
            get_snap_nr = np.where(f['snapshot'] == float(self.snap_nr))[0][0]
            r_min, r_max = f['d_c_min'].to_numpy()[get_snap_nr], f['d_c_max'].to_numpy()[get_snap_nr]

            t = QTable([flag, tot_mass, d_c, x, y, z, redshift, radius, stellar_mass, sfr,],
            names=("flag", "tot_mass", "d_c", "x", "y", "z", "redshift", "twice_half_mass_rad", "stellar_mass_in_2half_rad", "sfr_in_2half_rad"),
            meta={'SnapnNo': f"{self.snap_nr}",
                  'shellMin': f"{r_min:.2f} Mpc",
                  'shellMax': f"{r_max:.2f} Mpc" })

            t.write(lc_properties_file, overwrite=True)
            self.logger.info(f" File generated? {os.path.isfile(lc_properties_file)}")

        self.lightcone_shell_properties = lc_properties_file
        return

    def add_property(self, snap_nr, property_name: str = None, prop_unit = None, clobber=False, column_name: str = None, select_cloumn: bool = False, column_no: int = None) -> None:
        """Function to add a column of a quantity to the lightcone table

        Parameters
        ----------
        property_name :: str
            the name of the property to add
        prop_unit :: astropy.unit
            the unit of the property from IllustrisTNG that one is adding
        clobber :: bool
            decides wether to rerun the code
        column_name :: str
            the name of the column for the property being added
        """
        if property_name is None:
            raise TypeError(f"Please pass {property_name=}")

        if column_name is None:
            column_name = property_name

        self.get_filenames()

        # selected particles
        shell_ids = np.load(self.shell_ids)

        if not os.path.isfile(self.lightcone_shell_properties):
            self.get_lightcone_properties(snap_nr, clobber=clobber)
        else:
            t = QTable.read(self.lightcone_shell_properties, format='fits')

            # if the desired column does not exist, retrive it and add it
            if column_name not in t.colnames:
                prop = il.groupcat.loadSubhalos(self.sim_data_dir, self.snap_nr, fields=[property_name])[shell_ids]

                if select_cloumn:
                    if column_no is None:
                        raise TypeError("Please pass {column_no=}")
                    prop = prop[:, column_no]

                # add the column
                t.add_column(prop*prop_unit, name=column_name)
                t.write(self.lightcone_shell_properties, overwrite=True)
                self.logger.info(f"{column_name=} added to the table? {os.path.isfile(self.lightcone_shell_properties)}")
            # if the desired column does exists, but clobber set to True; rewrite it
            elif clobber:
                t.remove_column(column_name)
                prop = il.groupcat.loadSubhalos(self.sim_data_dir, self.snap_nr, fields=[property_name])[shell_ids]

                # add the column
                t.add_column(prop, name=column_name)
                t.write(self.lightcone_shell_properties, overwrite=True)
            else:
                self.logger.info(f"{column_name=} already exists. Set {clobber=} to True to rewrite.")
        return