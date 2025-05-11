### Installation
The package can be installed with:

`pip install light-gen`


You will need the following dependencies:

```
numpy
astropy
h5py
multiprocessing
glob
logging
seaborn
scipy
pandas
```



### Usage
Using the code to generate lightcones from IllustrisTNG simulations. This code is easily adaptable to other N-body simulations too. For instructions see the example notebooks provided in the `notebooks` folder and also see the template script.
The code needs you to point to the `sim_data_dir` folder, where all the snapshots and FoF and Subfind group catalogues are provided (based on hdf5). You can download them from the official TNG website [here](https://www.tng-project.org/data/). The `sim_data_dir` directory structure must be as follows.

```
IllustrisTNG/
 |-snapdir_98
 | |-snap_98.0.hdf5
 | |-snap_98.1.hdf5
 | `-snap_98.599.hdf5
 |-snapdir_97
 | |-snap_97.0.hdf5
 | |-snap_97.1.hdf5
 | `-snap_97.599.hdf5
 `-snapdir_78
   |-snap_78.0.hdf5
   |-snap_78.1.hdf5
   `-snap_78.599.hdf5
```
And for the group catalogues: 
```
IllustrisTNG/
 |-groups_98
 | |-fof_subhalo_tab_98.0.hdf5
 | |-fof_subhalo_tab_98.1.hdf5
 | `-fof_subhalo_tab_98.599.hdf5
 |-groups_97
 | |-fof_subhalo_tab_97.0.hdf5
 | |-fof_subhalo_tab_97.1.hdf5
 | `-fof_subhalo_tab_97.599.hdf5
 `-groups_96
   |-fof_subhalo_tab_96.0.hdf5
   |-fof_subhalo_tab_96.1.hdf5
   `-fof_subhalo_tab_96.599.hdf5
```

Where each `snapdir_XX` and `group_XX` refers to a single TNG snapshot, and each snapshot can be divided into multiple chunck files (going from 0 to 500).

### Examples


The follows notebooks run you through the main concepts and implementations of this code:
1. `01_remapping_IllustrisTNG_boxes`: introduced the remapping of cuboids to construct lightcones.
2. `02_get_lightcone_geometry`: implements the `get_geometry.py` to get the lightcone opening angles for an given observer at the edges of the lightcone.
3. `03_get_lightcone_and_central_satellities`: generates lightcone shell files for individual subshells (subhalos, halos, bhs, etc) and also classified subhalos into centrals and satellites
4. `04_applying_rotations_to_lightcone`: tutorial on rotations in 3D and how you can apply it to your lightcone to center the observer on the y-z plane (x-axis is Line of Sight)

Script: `main.py`

You can directly run `main.py` after passing the `sim_data_dir` with the appropriate data structure within as follows.

```
python3 main.py 0 1 gas /path/to/sim_data_dir
```
This would transform the coordinates for one chunk file (there are 600/snapshot) and save the transformed coordinates in your `out_data_dir` path. For `group` (FoF halos), `bh` (black holes), and `Subhalo` (from SUBFIND) the first two `int` arguments specify the entire snapshots (not chuncks as done for `gas` and `dm`).

### Authors

This package has been developed by Soumya Shreeram, with contributions from: Johan Comparat.

This package uses the `remap.py` and `list7.txt` files for cuboid remapping implementation, which is downloaded from http://mwhite.berkeley.edu/BoxRemap/. Additionally, see "Embedding realistic surveys in simulations through volume remapping", J. Carlson and M. White, ApJS 190(2010)311. (preprint available at arxiv:1003.3178).

### Contact
If you have any question, suggestion, or need help with the code (e.g., extending to other simulations with different data formats), don't hesitate to contact the author.

### Citation

If you use this code for you work, please cite the following two related papers: 
1. Shreeram et al. 2025a: https://www.aanda.org/articles/aa/abs/2025/05/aa52271-24/aa52271-24.html (also available or arXiv: https://arxiv.org/abs/2409.10397).
2. J. Carlson and M. White, ApJS 190(2010)311. (preprint available at arxiv:1003.3178).