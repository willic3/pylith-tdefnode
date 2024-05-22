### This directory contains the files needed to generate a merged spatial database for Vp/Vs/density used by PyLith, along with some other spatial database files.
Spatial databases are used by PyLith to specify variations in properties, including elastic properties and boundary conditions. The spatial database files already in this directory include:
1.  `impulse_amplitude_v2.spatialdb`: Specifies the impulse amplitudes when generating Green's functions (only used for PyLith v2.x).
2.  `mat_elastic_v2.spatialdb`: Specifies constant elastic properties when generating homogeneous Green's function for PyLith v2.x.
3.  `mat_elastic_v3.spatialdb`: Specifies constant elastic properties when generating homogeneous Green's function for PyLith v3+.
4.  `sliptime_v2.spatialdb`: Specifies the slip time when generating Greens' functions (only used for PyLith v2.x).

There are then 3 files used to produce a merged velocity model:
1.  `vps_3d_tomodd_mod_wujinaping.dat`:  High resolution velocity model used in the inner part of the domain.
2.  `liu_etal_2021_SRL_0.5x0.5_wrt_surface_VpVs.txt`:  Moderate resolution velocity model used outside of higher resolution region.
3.  `prem_1s.csv`:  PREM model used for the extreme outer boundaries of the domain.

To produce a merged velocity model, execute the Python script:

`./gen_velmodel.py`

This will produce either `merged_velmodel_v2.spatialdb` (if using Python 2) or `merged_velmodel_v3.spatialdb` (if using Python 3). Some VTK files will also be produced to visualize the velocity model. These velocity models are then used to provide properties to PyLith.
