This is the directory where we read the PyLith-generated GF and integrate them
to produce TDefnode-style GF. If the PyLith GF were generated in ../config,
the easiest thing to do to start with is to create a symbolic link to the
output directory:

ln -s ../config/output output

This simplifies all of the .cfg files used in this directory.

To generate the TDefnode GF, we use the py2def.py script:

./py2def.py p2d_az_homog.cfg

This will create a set of TDefnode GF in the pyhom_gf directory, as well
as information in the p2d_info/pyhom directory. These are VTK files that
may be viewed in Paraview. Note that the p2d_az .cfg files differ from
the others, since they specify 3d slip rather than shear slip.

To compare the PyLith GF with those from TDefnode, we use the read_def_gf.py
script:

./read_def_gf.py rdf_az_tdef11.cfg
./read_def_gf.py rdf_az_pyhom.cfg

This script reads TDefnode GF and generates a set of VTK files that may be
viewed in Paraview. The first command reads the native TDefnode GF (contained
in ../tdefnode/g11), and the second reads the PyLith equivalent GF (contained
in pyhom_gf).
