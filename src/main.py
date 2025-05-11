import numpy as np
from astropy import units as u

import multiprocessing as mp
from glob import glob
import sys
import os
import logging
logger = logging.getLogger(__name__)

# local imports for TNG
from generate_lightcones import Generate_lightcones
from remap_IllustrisTNG import Lightcone_IllustrisTNG

def generate_lightcones4all_observers(snapshot, chunck_file, observer_loc, particle_type, sim_data_dir,
                                      generating_lc=True,
                                      add_subhalo_property: bool = False,
                                      subhalo_properties=['SubhaloStellarPhotometrics',(u.mag), 'SubhaloKbandLuminosity',4]):
    """Function to generate the lightcones for all 8 observers

    Parameters
    ----------
    snapshot ::
        the snapshot number for which the lightcone shell is created
    chunck_file ::
        required only for gas particles, the subfile within the snapshot
    observer_loc :: int
        the location of the observer, ranges from 0-7
    particle_type :: str
        could be 'gas', 'bh'= black hole, 'star', 'halo' = clusters/groups, 'subhalo' = galaxies.
    generating_lc :: bool
        if set to true function performs the generation of the lightcone
    subhalo_properties :: list [str, astropy.units, str, int]
        if you would like to add a property to the subhalo properties file, set add_subhalo_property to True and input params to list

    Commands to run
    ---------------
    for ((i=0;i<=8000;i+=100)); do sbatch main.sh -t $i -y $(($i+100)) -P gas -O 0 -b 1; done
    -> Typically takes ~20-30 mins for each srun, with memory of ~60-70 GB for gas particles
    """

    if particle_type.lower() in ['gas', 'stars', 'star', 'dm'] and chunck_file is None:
        raise ValueError(f"Please pass snapshots in subfiles, i.e., chunck_file arg can not be {chunck_file=} ")

    obs_object = Generate_lightcones(observer_loc = observer_loc, snap_nr=snapshot, particle_type=particle_type, sim_data_dir=sim_data_dir, chunck_file=chunck_file)
    logger.info(f"{obs_object.snap_nr=}")

    if generating_lc:
        obs_object.produce_shells4observer(clobber=False)
    else:
        obs_object.get_lightcone_properties(clobber=True)
        if add_subhalo_property and particle_type.lower() in ['subhalo']:
            property_name, prop_unit, column_no, column_name = subhalo_properties
            obs_object.add_property(snap_nr=snapshot, property_name=property_name, prop_unit=prop_unit, column_no=column_no, column_name=column_name)
    logger.info(f"Hello, {snapshot=} is done from process {mp.current_process()}")
    return

def parallizeTNGruns(particle_type, snapshot_list, observer_loc, sim_data_dir):
    """Function to output the iteratable over which the script is parallized

    Notes
    -----
    Based on which function is set to run, you need to change the func argument 
    Possilbe functions so far are as follows:

    * set -b 1 for chunk_division
    
    chunk_division:
    ---------------
        1. generate_lightcones4all_observers
            this function generates the TNG lightcones for the gas, stellar and dm particles
    """
    iters = []
    if particle_type in ['gas', 'stars', 'dm']:
        for s in snapshot_list:
            obs_object = Lightcone_IllustrisTNG(particle_type = particle_type, snap_nr=s, sim_data_dir=sim_data_dir)
            obs_object.chunck_file_number = 0
            obs_object.get_filenames(bin_no=1, observer_loc=observer_loc)
            
            # parallelization by chunks
            chunck_file_list =  glob(f"{obs_object.sim_data_dir}/snapdir_0{s}/*.hdf5")

            for c_file in chunck_file_list:
                iters.append((s, c_file, observer_loc, particle_type, sim_data_dir))

            func = generate_lightcones4all_observers
    
    # e.g., blackholes, Subhalos and halos -> only snapshot division, no subshell info needed
    else:            
        for s in snapshot_list:
            iters.append((s, None, observer_loc, particle_type, sim_data_dir))
        func = generate_lightcones4all_observers
    return iters, func


if __name__ == '__main__':
    # the snapshots I parallize for going upto z=0.3 or los distance = 1231 Mpc
    # change if you build a lightcone with different preferences
    snapshot_list = np.arange(78, 97).astype(str)

    # multiprocessing
    pool = mp.Pool(100)
    with pool as p:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        
        # input params
        start_iter, end_iter = int(sys.argv[1]), int(sys.argv[2]) # 1st and 2nd argument
        particle_type = str(sys.argv[3]) # 3rd argument
        observer_loc = int(sys.argv[4]) # 4th argument
        sim_data_dir = str(sys.argv[5]) # 5th argument
        
        logger.info(f"{start_iter=}, {end_iter=}, {observer_loc=}, {particle_type=}")

        iters, func = parallizeTNGruns(particle_type, snapshot_list, observer_loc, sim_data_dir)
        logger.info(f"{len(iters)=}")
        p.starmap(func, iters[start_iter: end_iter])
    pool.close()
    pool.join()