This directory contains the files necessary to create a TDEFNODE-based mesh
for computing Green's functions.  The steps are as follows:

1.  Go into the geometry directory, and run the 'create_defnode_quads.py'
    script. This script automatically reads the 'create_defnode_quads.cfg'
    configuration script, which in turn specifies the .nod file to use
    and several other parameters. You can just run the script by doing:

    create_tdefnode_quads.py

    After running the script, there will be several VTK files that can be
    viewed in Paraview, as well as a number of journal files with
    Cubit/Trelis commands.

2.  Run Cubit/Trelis, and execute each of the *_master.jou files. This will
    create a set of .cub files that can be read by Cubit/Trelis.

3.  Move back into the 'mesh' directory (this directory), and then within
    Cubit/Trelis execute each of the *_geometry.jou files. This will create
    a Cubit geometry file that can then be used for meshing. Note that it
    may be necessary to adjust some of the parameters to get the fault to
    properly intersect your bounding box (e.g., xdim, ydim, etc.).

4.  Create an initial mesh by running Cubit/Trelis with the *_var_lev1.jou
    files. This will create a coarse mesh onto which we can put sizing
    function information. Note that these journal files will call the
    *_bc.jou files, which assign blocks and define nodesets that are
    needed by PyLith. If anything is changed in the geometry file, the
    other files will probably need to be updated.

5.  Run the create_cell_size_3d.py script to generate sizing function
    information, e.g.:

    create_cell_size_3d.py anning_zem_ccs3.cfg

6.  After running the sizing function code you can look at the sizing
    function info by opening the .exo file in ParaView. Make sure you
    click the 'cell_size' button for arrays to show. It is sometimes
    necessary to go back and forth with the parameters in the .cfg
    file to get a nice cell size distribution.

7.  Once you have the cell size the way you want it, you can create
    a graded mesh using the *_graded.jou files in Cubit/Trelis. This
    will read in the sizing function information from the earlier
    mesh and use it to generate a new mesh. Once you've done this
    you should have a nicely graded mesh to use for generating
    Green's functions. Note that it is generally necessary to look
    at the mesh quality within Cubit (use the Condition Number metric
    for all volumes). You should try to have a maximum condition number
    of 2 or less. Slightly more is OK, but will affect your solution
    accuracy and computation time.

NOTE:  I have now gone through this procedure for all 6 faults. Note that
       I had to change the geometry slightly for lijiang, because the
       original inner box was cutting off a sliver of the fault.