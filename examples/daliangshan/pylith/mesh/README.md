### This directory contains the files necessary to create a TDEFNODE-based mesh for computing Green's functions. Note that the Cubit meshing software is required to generate the meshes.

The steps are as follows:

1.  Go into the geometry directory, and run the `pytdef_create_defnode_quads` script that you installed previously. This script automatically reads the `pytdef_create_defnode_quads.cfg` configuration script, which in turn specifies the py01.nod file in the `data` directory that defines the TDefnode fault geometries. You can just run the script by doing:

    `pytdef_create_tdefnode_quads`

    After running the script, there will be several VTK files that can be viewed in Paraview, as well as a number of journal files with Cubit commands.

2.  Run Cubit, and execute each of the `*_master.jou` files. This will create a set of `.cub` files that can be read by Cubit.

3.  Move back into the `mesh` directory (this directory), and then within Cubit execute each of the `*_geometry.jou` files. This will create a Cubit geometry file for each fault that can then be used for meshing. Note that it may be necessary to adjust some of the parameters to get the fault to properly intersect your bounding box (e.g., xdim, ydim, etc.).

4.  Create an initial mesh by running Cubit with the `*_var_lev1.jou` files. This will create a coarse mesh onto which we can put sizing function information. Note that these journal files will call the `*_bc.jou` files, which assign blocks and define nodesets that are needed by PyLith. If anything is changed in the geometry file, the other files will probably need to be updated.

5.  Run the `pytdef_create_cell_size` script to generate sizing function information, e.g.:

    `pytdef_create_cell_size anning_zem_ccs3.cfg`

6.  After running the sizing function code you can look at the sizing function info by opening the `.exo` file in ParaView. Make sure you click the 'cell_size' button for arrays to show. It is sometimes necessary to go back and forth with the parameters in the `.cfg` file to get a nice cell size distribution.

7.  Once you have the cell size the way you want it, you can create a graded mesh using the `*_graded.jou` files in Cubit. This will read in the sizing function information from the earlier mesh and use it to generate a new mesh. Once you've done this you should have a nicely graded mesh to use for generating Green's functions. Note that it is generally necessary to look at the mesh quality within Cubit (use the Condition Number metric for tets within all volumes). You should try to have a maximum condition number of 2 or less. Slightly more is OK, but will affect your solution accuracy and computation time.

NOTE:  I have now gone through this procedure for all 6 faults. Note that I had to change the geometry slightly for lijiang, because the original inner box was cutting off a sliver of the fault.
