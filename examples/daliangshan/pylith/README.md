# This is the top-level directory for setting up PyLith Green's functions to use with TDefnode.

The steps are as follows:

1.  Create the meshes. To do that, change to the `mesh` directory and follow the instructions in the README there.
2.  Set up the `data` directory. This directory contains the locations of the GNSS sites, the TDefnode `.nod` file describing the TDefnode fault geometries, and a directory containing Green's functions produced by TDefnode. Change to the `data` directory and follow the instructions in the README there.
3.  Set up the spatial database defining the merged seismic velocity model used for this project. Change to the `spatialdb` directory and follow the instructions in the README there.
4.  Run the PyLith simulations to generate PyLith Green's functions. If you are using PyLith v2.x, change to the `config-v2` directory and follow the instructions in the README there. If you are using PyLith v3+, change to the `config-v3` directory and follow the instructions in the README there.
5.  Integrate the PyLith Greens' functions to obtain a set of Green's functions that may be used with TDefnode. Change to the `greensfns` directory and follow the instructions in the README there.
6.  To compute the shear modulus values to use when computing seismic moment, you can use either the `moment-v2` directory (if using Python/PyLith v2.x) or the `moment-v3` directory if using Python/PyLith v3+. Change to the appropriate directory and follow the instructions in the README there.

Once you have finished the steps above you will have generated a set of Green's functions that can be used as drop-in replacements for those used by TDefnode.
