This directory contains files needed to generate a merged spatial
database for Vp/Vs/density used by PyLith, along with a simple
spatial database file for constant properties (mat_elastic.spatialdb).
The constant properties yield values of lambda = mu = 30 GPa.
The Python script (gen_velmodel.py) creates a velocity model where the
values in the center are from Liu et al. (2021, SRL), with density values
computed from the relation defined by equation (1) of Brocher (2005, BSSA).
Outside of this region the values are defined by PREM. There is a 'buffer'
region surrounding the inner volume over which linear 3D interpolation is
performed to merge the Liu et al. results with PREM. RBF interpolation is
another option (currently commented out in the code), but the linear
interpolation results seem better at this point.

To generate the merged velocity model, simply do:

./gen_velmodel.py

This will create the merged_velmodel.spatialdb spatial database file, along
with some VTK files that may be viewed with Paraview. The final spatialdb
files that should be present in this directory are:

impulse_amplitude.spatialdb:    Used when generating slip impulses for PyLith
                                Green's functions.
mat_elastic.spatialdb:          Used to define material properties for
                                homogeneous property Green's functions.
merged_velmodel.spatialdb:      Used to define material properties for
                                heterogeneous property Green's functions.
sliptime.spatialdb:             Needed by PyLith when generating Green's functions.
