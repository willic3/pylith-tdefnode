### This is the directory where we read the PyLith-generated Green's functions and integrate them to produce TDefnode-style Green's functions.

Since we make use of the PyLith GF in this directory, the easiest thing to do is to create a symbolic link to the directory containing the PyLith Green's functions. For example, if you created the Green's functions using PyLith version 3+, you could do the following:

```
ln -s ../config-v3/output output
```

Doing this will simplify all of the `.cfg` files used in this directory. In particular, this takes care of the problem of which config directory contains your Green's functions (`config-v2` or `config-v3`).

To generate the TDefnode GF, we use the `pytdef_py2def` script you installed previously:

```
pytdef_py2def p2d_az_homog.cfg
```

This will create a set of TDefnode GF in the `pyhom_gf` directory, as well as information in the `p2d_info/pyhom` directory. Note that you must have previously generated the appropriate Green's functions using PyLith. The command above will generate the Green's functions for the Anninghe-Zemuhe Fault System assuming homogeneous (constant) material properties. The files in the `phom_gf` directory are the replacement Green's functions for those produced by TDefnode. The files in the `p2d_info/phom` directory are VTK files that may be viewed in Paraview. Note that the `p2d_az*.cfg` files differ from the others, since they specify 3d slip rather than shear slip (see TDefnode documentation).

To compare the PyLith GF with those from TDefnode, we use the `pytdef_read_def_gf` script you installed previously:

```
pytdef_read_def_gf rdf_az_tdef.cfg
pytdef_read_def_gf rdf_az_pyhom.cfg
```

This script reads TDefnode GF and generates a set of VTK files that may be viewed in Paraview. The first command reads the native TDefnode GF (contained in `../data/d22`), and the second reads the PyLith equivalent GF (contained in `pyhom_gf`). When opening these files in Paraview, select all files with the same prefix at once (this is an option in Paraview). There will be one set for impulses and another for responses. The applied impulses may be viewed by coloring the fault by `fault_slip`, and the associated displacements at the GNSS sites may be viewed using the `Glyph` filter. You can use the `Arrow` glyph, and then use the desired response array for orientation and scaling. The `*_response_e` array is the response due to impulses applied in the east direction, etc. If you load both sets of responses from the commands above, you can compare the responses for the two models. You can cycle through the impulses by using the movie controls in Paraview.

There is another script to explicitly compare two sets of GF:

```
pytdef_compare_def_gf cdf_az_pyhet_minus_pyhom.cfg
```

This script will compare the differences between two sets of GF using the VTK files produced by `pytdef_read_def_gf`. The output will consist of a set of VTK files containing the differences, along with a histogram of the differences. Note that you must have run `pytdef_read_def_gf` on both of the sets of GF prior to using this script.

**NOTE:** If you are using the PyLith 2.x binary, the tkinter package does not appear to be functional. This will result in an error when attempting to run the `pytdef_compare_def_gf` script.
