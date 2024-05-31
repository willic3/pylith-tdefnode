### This is the directory where we compute shear modulus values for TDefnode faults.

Note that this directory is specific to Python/PyLith v2.x.

Since we typically use spatially varying material properties when using PyLith, it is no longer appropriate to assume a constant shear modulus when computing seismic moment. The `pytdef_get_tdef_moment` script may be used to get the effective shear modulus on the fault. The script samples the seismic velocity model spatial database on either side of the fault. Differences in shear moduls across the fault are accounted for used the effective shear modulus, as defined by Wu and Chen (2003) to account for differences across the fault. The script produces a modified TDefnode output file with the varying shear modulus, as well as a VTK file for visualization.

The parameters for the script are in the `pytdef_get_tdef_moment.cfg` file (common parameters for all faults), and the parameters for each fault are in the `gtm_f*.cfg` files. Run the script as follows:

```
pytdef_get_tdef_moment gtm_f001.cfg
```
