### Directories, files and descriptions

1. `tools` 
    └── `__init__.py`: initializes files within directory <br>
    └── `plotting.py`: some useful plotting routines that are used in the example notebooks <br>
    └── `remap.py`: cuboid remapping implementation, which is downloaded from http://mwhite.berkeley.edu/BoxRemap/ <br>
    └── `vec3.py`: part of the boxremap files. <br>
    └── `list7.txt`: file with possible side lengths L1 and L2 for the remapped cuboid, in units of the length Lbox of the original simulation cube<br>
    └── `illustris_python` python scripts for using IllustrisTNG. Downloaded from https://github.com/illustristng/illustris_python. 
    
### Files
1. `util` 
    contains all the helper functions

3. `remap_IllustrisTNG`
    contains the class to generate lightcones from the hydro-dynamical simulation outputs of IllustrisTNG

4. `get_geometry`
    contains the class `MakeCuboid` and `Geometry`, for obtaining the properties for getting intersections between a cuboid and spehere

6. `generate_lightcones` 
    contains the class to generate lightcones for gas particles, stellar particles, galaxies, and halos

6. `main.py` 
    example script showing how the run the code