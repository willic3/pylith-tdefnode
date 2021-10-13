#!/usr/bin/env python

## @file create_cell_size_3d.py

## @brief Python application to create an Exodus sizing function based
## on three different functions (ellipsoidal, RBF interpolation, and
## gradient of strain energy).

import math
import numpy
import netCDF4
import h5py
import scipy.interpolate
import scipy.spatial
import scipy.spatial.distance
import sys
import os

from pyre.applications.Script import Script as Application

class CreateCellSize3D(Application):
    """
    Python application to create an Exodus sizing function based
    on three different functions (ellipsoidal, RBF interpolation, and
    gradient of strain energy).
    """

    ## \b Properties
    ## @li \b exodus_input_file Exodus II input file.
    ## @li \b size_func_fracs SF contributions from ellipse, gradient, and RBF.
    ## @li \b ell_min_size Minimum cell size for ellipse sizing function.
    ## @li \b ell_max_size Maximum cell size for ellipse sizing function.
    ## @li \b ell_center_coords (x,y,z) coordinates of ellipsoid center.
    ## @li \b ell_axis_lengths (x,y,z) lengths of ellipsoid axes.
    ## @li \b ell_x_dir_cos Direction cosines for x-axis rotation.
    ## @li \b ell_y_dir_cos Direction cosines for y-axis rotation.
    ## @li \b ell_z_dir_cos Direction cosines for z-axis rotation.
    ## @li \b ell_sf_exp Ellipsoid sizing function exponent.
    ## @li \b grad_hdf5_file HDF5 file used to estimate strain energy gradient.
    ## @li \b grad_min_size Minimum cell size for gradient sizing function.
    ## @li \b grad_max_size Maximum cell size for gradient sizing function.
    ## @li \b grad_fraction_min Fraction of nodes with min cell size for grad SF.
    ## @li \b grad_function_type Type of gradient sizing function to use.
    ## @li \b grad_function_scaling Type of scaling for gradient sizing function.
    ## @li \b rbf_const_ns_names Nodeset names to assign constant sizes.
    ## @li \b rbf_const_ns_sizes Sizes to assign to each nodeset.
    ## @li \b rbf_const_ns_sampfreq Sampling frequency for constant size nodesets.
    ## @li \b rbf_var_ns_names Nodeset names with linearly varying sizes.
    ## @li \b rbf_var_ns_edge1_names Nodeset names for edge 1 of linearly varying sizes.
    ## @li \b rbf_var_ns_edge1_sizes Sizes along edge 1.
    ## @li \b rbf_var_ns_edge2_names Nodeset names for edge 2 of linearly varying sizes.
    ## @li \b rbf_var_ns_edge2_sizes Sizes along edge 2.
    ## @li \b rbf_var_ns_sampfreq Sampling frequency for linearly-varying nodesets.
    ## @li \b rbf_size_scale Factor by which to scale sizes from point file.
    ## @li \b rbf_num_mesh_slices Number of slices to use when applying RBF.
    ## @li \b rbf_smoothing Smoothing factor for RBF SF.
    ## @li \b rbf_epsilon Epsilon factor for RBF SF (only applies to gaussian or multiquadric).
    ## @li \b rbf_type Type of RBF to use for RBF SF.
    ## @li \b print_incr Increment for which to print current cell number.

    import pyre.inventory

    exodusInputFile = pyre.inventory.str("exodus_input_file", default="mesh.exo")
    exodusInputFile.meta['tip'] = "Exodus II input file."

    sizeFuncFracs = pyre.inventory.list("size_func_fracs", default=[0.33, 0.33, 0.33])
    sizeFuncFracs.meta['tip'] = "SF contributions from ellipse, RBF, and gradient."

    ellMinSize = pyre.inventory.float("ell_min_size", default=1000.0)
    ellMinSize.meta['tip'] = "Minimum cell size for ellipse sizing function."

    ellMaxSize = pyre.inventory.float("ell_max_size", default=10000.0)
    ellMaxSize.meta['tip'] = "Maximum cell size for ellipse sizing function."

    ellCenterCoords = pyre.inventory.list("ell_center_coords", default=[0.0, 0.0, 0.0])
    ellCenterCoords.meta['tip'] = "(x,y,z) coordinates of ellipsoid center."

    ellAxisLengths = pyre.inventory.list("ell_axis_lengths", default=[5000.0, 5000.0, 5000.0])
    ellAxisLengths.meta['tip'] = "(x,y,z) lengths of ellipsoid axes."

    ellXDirCos = pyre.inventory.list("ell_x_dir_cos", default=[1.0, 0.0, 0.0])
    ellXDirCos.meta['tip'] = "Direction cosines for x-axis rotation."

    ellYDirCos = pyre.inventory.list("ell_y_dir_cos", default=[0.0, 1.0, 0.0])
    ellYDirCos.meta['tip'] = "Direction cosines for y-axis rotation."

    ellZDirCos = pyre.inventory.list("ell_z_dir_cos", default=[0.0, 0.0, 1.0])
    ellZDirCos.meta['tip'] = "Direction cosines for z-axis rotation."

    ellSfExp = pyre.inventory.float("ell_sf_exp", default=1.3)
    ellSfExp.meta['tip'] = "Ellipsoid sizing function exponent."

    gradHdf5File = pyre.inventory.str("grad_hdf5_file", default="statevars.h5")
    gradHdf5File.meta['tip'] = "HDF5 file to compute strain energy gradients."

    gradMinSize = pyre.inventory.float("grad_min_size", default=1000.0)
    gradMinSize.meta['tip'] = "Minimum cell size for gradient sizing function."

    gradMaxSize = pyre.inventory.float("grad_max_size", default=10000.0)
    gradMaxSize.meta['tip'] = "Maximum cell size for gradient sizing function."

    gradFractionMin = pyre.inventory.float("grad_fraction_min", default=0.15)
    gradFractionMin.meta['tip'] = "Fraction of nodes with min cell size for gradient SF."

    gradFunctionType = pyre.inventory.str("grad_function_type", default="sqrt_strain_energy",
                                          validator=pyre.inventory.choice(["strain_energy", "sqrt_strain_energy",
                                                                           "strain_norm", "stress_norm"]))
    gradFunctionType.meta['tip'] = "Type of gradient sizing function to use."

    gradFunctionScaling = pyre.inventory.str("grad_function_scaling", default="log",
                                             validator=pyre.inventory.choice(["linear", "log", "root2", "root3",
                                                                              "root8", "root16"]))
    gradFunctionScaling.meta['tip'] = "Type of gradient sizing function scaling."

    rbfConstNsNames = pyre.inventory.list("rbf_const_ns_names", default=["ns1", "ns2"])
    rbfConstNsNames.meta['tip'] = "Nodeset names to assign constant sizes for RBF interpolation."

    rbfConstNsSizes = pyre.inventory.list("rbf_const_ns_sizes", default=[60000.0, 1000.0])
    rbfConstNsSizes.meta['tip'] = "Sizes to assign to each nodeset for RBF interpolation."

    rbfConstNsSampfreq = pyre.inventory.list("rbf_const_ns_sampfreq", default=[1, 1])
    rbfConstNsSampfreq.meta['tip'] = "Sampling frequency for constant size nodesets for RBF interpolation."

    rbfVarNsNames = pyre.inventory.list("rbf_var_ns_names", default=["ns3", "ns4"])
    rbfVarNsNames.meta['tip'] = "Nodeset names with linearly varying sizes for RBF interpolation."

    rbfVarNsEdge1Names = pyre.inventory.list("rbf_var_ns_edge1_names", default=["nsedge1", "nsedge2"])
    rbfVarNsEdge1Names.meta['tip'] = "Nodeset names for edge 1 of linearly varying sizes."

    rbfVarNsEdge1Sizes = pyre.inventory.list("rbf_var_ns_edge1_sizes", default=[10000.0, 5000.0])
    rbfVarNsEdge1Sizes.meta['tip'] = "Sizes along edge 1 for RBF interpolation."

    rbfVarNsEdge2Names = pyre.inventory.list("rbf_var_ns_edge2_names", default=["nsedge3", "nsedge4"])
    rbfVarNsEdge2Names.meta['tip'] = "Nodeset names for edge 2 of linearly varying sizes."

    rbfVarNsEdge2Sizes = pyre.inventory.list("rbf_var_ns_edge2_sizes", default=[1000.0, 500.0])
    rbfVarNsEdge2Sizes.meta['tip'] = "Sizes along edge 2 for RBF interpolation."

    rbfVarNsSampfreq = pyre.inventory.list("rbf_var_ns_sampfreq", default=[1, 1])
    rbfVarNsSampfreq.meta['tip'] = "Sampling frequency for linearly-varying nodesets for RBF interpolation."

    rbfSizeScale = pyre.inventory.float("rbf_size_scale", default=1.0)
    rbfSizeScale.meta['tip'] = "Scaling factor to apply to sizes from point file."

    rbfSmoothing = pyre.inventory.float("rbf_smoothing", default=0.1)
    rbfSmoothing.meta['tip'] = "Smoothing factor for RBF sizing function."

    rbfEpsilon = pyre.inventory.float("rbf_epsilon", default=0.1)
    rbfEpsilon.meta['tip'] = "Epsilon factor for RBF sizing function (only applies to gaussian or multiquadric)."

    rbfNumMeshSlices = pyre.inventory.int("rbf_num_mesh_slices", default=1)
    rbfNumMeshSlices.meta['tip'] = "Number of slices to use when applying RBF (to avoid memory errors)."

    rbfType = pyre.inventory.str("rbf_type", default="linear",
                                 validator=pyre.inventory.choice(["multiquadric", "inverse", "gaussian",
                                                                  "linear", "cubic", "quintic", "thin_plate"]))
    rbfType.meta['tip'] = "Type of RBF to use."
  
    printIncr = pyre.inventory.int("print_incr", default=10000)
    printIncr.meta['tip'] = "Increment for which to print current cell number."
  

    # PUBLIC METHODS /////////////////////////////////////////////////////

    def __init__(self, name="create_cell_size_3d"):
        Application.__init__(self, name)

        self.numSfTypes = 0
        self.exodus = None
        self.spaceDim = 3
        self.meshCoords = None
        self.numMeshVerts = 0
        self.numNodeSets = 0
        self.ellSizeFunc = None
        self.gradSizeFunc = None
        self.rbfSizeFunc = None
        self.totSizeFunc = None
        self.useEllSizeFunc = False
        self.useGradSizeFunc = False
        self.useRbfSizeFunc = False
        self.cellType = None
        self.vertsPerCell = 0
        
        self.vertVolumes = None
        
        self.largeExodus = True

        self.tetVertArr = numpy.ones((4,4), dtype=numpy.float64)

        return


    def main(self):
        # import pdb
        # pdb.set_trace()

        funcFracs = [float(i) for i in self.sizeFuncFracs]
        self.sizeFuncRatios = numpy.array(funcFracs, dtype=numpy.float64)
        total = self.sizeFuncRatios.sum()
        self.sizeFuncRatios /= total
        self.numSfTypes = numpy.count_nonzero(self.sizeFuncRatios)

        self._readMesh()
        self._computeVolumes()

        self.totSizeFunc = numpy.zeros(self.numMeshVerts, dtype=numpy.float64)
    
        if (self.sizeFuncRatios[0] != 0):
            self.useEllSizeFunc = True
            self._calcEllSizeFunc()
            self.totSizeFunc += self.sizeFuncRatios[0] * self.ellSizeFunc
        if (self.sizeFuncRatios[1] != 0):
            self.useGradSizeFunc = True
            if (self.cellType == 'tet'):
                self._calcGradSizeFuncTet()
            else:
                self._calcGradSizeFuncTri()
            self.totSizeFunc += self.sizeFuncRatios[1] * self.gradSizeFunc
        if (self.sizeFuncRatios[2] != 0):
            self.useRbfSizeFunc = True
            self._calcRbfSizeFunc()
            self.totSizeFunc += self.sizeFuncRatios[2] * self.rbfSizeFunc

        self._writeSizeFunc()

        return


    # PRIVATE METHODS ////////////////////////////////////////////////////

    def _configure(self):
        """
        Setup members using inventory.
        """

        # pdb.set_trace()
        Application._configure(self)
        self.rbfConstNsSizes = [float(i) for i in self.rbfConstNsSizes]
        self.rbfConstNsSampfreq = [int(i) for i in self.rbfConstNsSampfreq]
        self.rbfVarNsEdge1Sizes = [float(i) for i in self.rbfVarNsEdge1Sizes]
        self.rbfVarNsEdge2Sizes = [float(i) for i in self.rbfVarNsEdge2Sizes]
        self.rbfVarNsSampfreq = [int(i) for i in self.rbfVarNsSampfreq]

        return


    def _computeVolumes(self):
        """
        Compute tetrahedral volumes and nodal contributions.
        """

        print('Computing tetrahedral volumes:')
        print('')

        cellTets = numpy.ones((self.numMeshCells, 4, 4), dtype=numpy.float64)
        cellVerts = self.meshCoords[self.connect,:]
        cellTets[:,1:4,:] = numpy.swapaxes(cellVerts, 1,2)
        cellVolumes = numpy.fabs(numpy.linalg.det(cellTets)/6.0)
        self.cellVolumes = cellVolumes.copy()
        self.vertVolumes = numpy.zeros(self.numMeshVerts, dtype=numpy.float64)
        cellVolumes *= 0.25
        for cellNum in range(self.numMeshCells):
            if (cellNum % self.printIncr == 0):
                print("    Working on cell # %d:" % cellNum)
                sys.stdout.flush()
            cellVerts = self.connect[cellNum,:]
            self.vertVolumes[cellVerts] += cellVolumes[cellNum]

        return

      
    def _readMesh(self):
        """
        Function to read coordinates from Exodus II file.
        """

        print("Reading Exodus II file:")
        print("")
        sys.stdout.flush()
    
        self.exodus = netCDF4.Dataset(self.exodusInputFile, 'a')
        try:
            x = self.exodus.variables['coordx'][:]
            y = self.exodus.variables['coordy'][:]
            z = self.exodus.variables['coordz'][:]
            self.meshCoords = numpy.column_stack((x,y,z))
        except:
            self.largeExodus = False
            self.meshCoords = self.exodus.variables['coord'][:].transpose()

        self.numMeshVerts = self.meshCoords.shape[0]
        spaceDim = self.meshCoords.shape[1]
        self.numNodeSets = len(self.exodus.dimensions['num_node_sets'])

        self.numVertsPerCell = len(self.exodus.dimensions['num_nod_per_el1'])
        if (self.numVertsPerCell == 4):
            self.cellType = 'tet'
        elif (self.numVertsPerCell == 3):
            self.cellType = 'tri'
        else:
            msg = 'Invalid cell type with %d vertices per cell.' % self.numVertsPerCell
            raise ValueError(msg)
      
        if (spaceDim != self.spaceDim):
            msg = "Spatial dimension of mesh must be equal to 3."
            raise ValueError(msg)

        numElemBlocks = len(self.exodus.dimensions['num_el_blk'])
        self.connect = self.exodus.variables['connect1'][:]
        if (numElemBlocks != 1):
            msg = "Current code is set up to only use one element block."
            raise ValueError(msg)

        self.connect -= 1
        self.numMeshCells = self.connect.shape[0]

        return


    def _triArea(self, verts):
        """
        Function to compute the area of a triangle.
        """
        v1 = verts[1,:] - verts[0,:]
        v2 = verts[2,:] - verts[0,:]
        area = 0.5 * math.fabs(numpy.cross(v1, v2))

        return area

  
    def _tetVolume(self, verts):
        """
        Function to compute the volume of a tet.
        """

        self.tetVertArr[1:4,:] = verts
        volume = math.fabs(numpy.linalg.det(self.tetVertArr)/6.0)

        return volume

  
    def _calcEllSizeFunc(self):
        """
        Function to compute ellipsoidal sizing function.
        """

        print("Computing ellipsoidal sizing function:")
        sys.stdout.flush()
    
        # Get axis lengths and compute squares.
        aEllps = float(self.ellAxisLengths[0])
        bEllps = float(self.ellAxisLengths[1])
        cEllps = float(self.ellAxisLengths[2])

        aSq = aEllps * aEllps
        bSq = bEllps * bEllps
        cSq = cEllps * cEllps

        # Shift and rotate coordinates.
        xCent = float(self.ellCenterCoords[0])
        yCent = float(self.ellCenterCoords[1])
        zCent = float(self.ellCenterCoords[2])
        coordsCent = numpy.array([xCent, yCent, zCent], dtype=numpy.float64)

        coordsShift = self.meshCoords - coordsCent

        xDirCos = numpy.array([float(i) for i in self.ellXDirCos], dtype=numpy.float64)
        yDirCos = numpy.array([float(i) for i in self.ellYDirCos], dtype=numpy.float64)
        zDirCos = numpy.array([float(i) for i in self.ellZDirCos], dtype=numpy.float64)
        coordsRotX = numpy.dot(coordsShift, xDirCos)
        coordsRotY = numpy.dot(coordsShift, yDirCos)
        coordsRotZ = numpy.dot(coordsShift, zDirCos)

        # Compute sizing function and write out some info.
        ellipse = numpy.maximum(numpy.sqrt(coordsRotX * coordsRotX/aSq +
                                           coordsRotY * coordsRotY/bSq + 
                                           coordsRotZ * coordsRotZ/cSq), 1.0)**self.ellSfExp
        self.ellSizeFunc = numpy.minimum(self.ellMinSize * ellipse, self.ellMaxSize)

        meanEllipseF = numpy.mean(self.ellSizeFunc)
        wtMeanEllipseF = numpy.average(self.ellSizeFunc, weights=self.vertVolumes)
        minEllipseF = numpy.amin(self.ellSizeFunc)
        maxEllipseF = numpy.amax(self.ellSizeFunc)

        print("  Mean ellipsoidal sizing function:                     %g" % meanEllipseF)
        print("  Volume-weighted mean ellipsoidal sizing function:     %g" % wtMeanEllipseF)
        print("  Minimum ellipsoidal sizing function:                  %g" % minEllipseF)
        print("  Maximum ellipsoidal sizing function:                  %g" % maxEllipseF)
        print("")
        sys.stdout.flush()

        return

  
    def _tensorInnerProd(self, tensor1, tensor2):
        """
        Function to compute tensor inner product for two tensors represented as
        6-vectors.
        """
        product = tensor1[:,0] * tensor2[:,0] + \
                  tensor1[:,1] * tensor2[:,1] + \
                  tensor1[:,2] * tensor2[:,2] + \
                  2.0 * (tensor1[:,3] * tensor2[:,3] + \
                         tensor1[:,4] * tensor2[:,4] + \
                         tensor1[:,5] * tensor2[:,5])

        return product

  
    def _calcGradSizeFuncTet(self):
        """
        Function to compute gradient sizing function for tetrahedral cells.
        """

        print("Calculating gradient sizing function:")
        sys.stdout.flush()

        # Read HDF5 file.
        data = h5py.File(self.gradHdf5File, "r")
        coordsHdf = data['geometry/vertices'][:]
        numVerts = coordsHdf.shape[0]
        spaceDim = coordsHdf.shape[1]
        if (spaceDim != self.spaceDim):
            msg = "Spatial dimension of mesh must be equal to 3."
            raise ValueError(msg)

        cellConnect = numpy.array(data['topology/cells'][:], dtype=numpy.int64)
        numCells = cellConnect.shape[0]
        numVertsPerCell = cellConnect.shape[1]
        if (numVertsPerCell !=4):
            msg = "Only tetrahedral cells may be used."
            raise ValueError(msg)

        fields = data['cell_fields']
        strain = fields['total_strain'][0,:,:]
        stress = fields['stress'][0,:,:]

        # Compute function for which to take gradient.
        if (self.gradFunctionType == 'strain_energy'):
            functionVals = self._tensorInnerProd(strain, stress)
        elif (self.gradFunctionType == 'sqrt_strain_energy'):
            functionVals = numpy.sqrt(self._tensorInnerProd(strain, stress))
        elif (self.gradFunctionType == 'strain_norm'):
            functionVals = numpy.sqrt(self._tensorInnerProd(strain, strain))
        elif (self.gradFunctionType == 'stress_norm'):
            functionVals = numpy.sqrt(self._tensorInnerProd(stress, stress))

        # Loop over cells and compute volumes and function values at vertices.
        functionWtVert = numpy.zeros(numVerts, dtype=numpy.float64)
        volVert = numpy.zeros(numVerts, dtype=numpy.float64)
        print("")
        print("  Computing function values at vertices:")
        for cellNum in range(numCells):
            if (cellNum % self.printIncr == 0):
                print("    Working on cell # %d:" % cellNum)
                sys.stdout.flush()
            cellVerts = cellConnect[cellNum,:]
            cellCoords = coordsHdf[cellVerts,:].transpose()
            cellVol = 0.25 * self._tetVolume(cellCoords)
            volVert[cellVerts] += cellVol
            functionWtVert[cellVerts] += cellVol * functionVals[cellNum]

        # Divide by vertex volumes to get vertex function values.
        functionVert = functionWtVert/volVert

        # Loop over cells and compute function gradient for each cell and vertex.
        gradCell = numpy.zeros(numCells, dtype=numpy.float64)
        gradWtVert = numpy.zeros(numVerts, dtype=numpy.float64)

        print("")
        print("  Computing function gradient for cells and vertices:")
        for cellNum in range(numCells):
            if (cellNum % self.printIncr == 0):
                print("    Working on cell # %d:" % cellNum)
                sys.stdout.flush()
            cellVerts = cellConnect[cellNum,:]
            refCoord = coordsHdf[cellVerts[0]]
            addCoords = coordsHdf[cellVerts[1:4]]
            diffCoords = addCoords - refCoord
            refFunction = functionVert[cellVerts[0]]
            addFunction = functionVert[cellVerts[1:4]]
            diffFunction = addFunction - refFunction
            gradient = numpy.linalg.solve(diffCoords, diffFunction)
            gradNorm = numpy.linalg.norm(gradient)
            gradCell[cellNum] = gradNorm
            gradWtVert[cellVerts] += volVert[cellVerts] * gradCell[cellNum]

        gradVert = gradWtVert/volVert

        # Print out gradient info for cells and vertices.
        meanGradCell = numpy.mean(gradCell)
        minGradCell = numpy.amin(gradCell)
        maxGradCell = numpy.amax(gradCell)
        print("")
        print("  Mean function gradient for cells:                     %g" % meanGradCell)
        print("  Minimum function gradient for cells:                  %g" % minGradCell)
        print("  Maximum function gradient for cells:                  %g" % maxGradCell)
        meanGradVert = numpy.mean(gradVert)
        wtMeanGradVert = numpy.average(gradVert, weights=volVert)
        minGradVert = numpy.amin(gradVert)
        maxGradVert = numpy.amax(gradVert)
        print("")
        print("  Mean function gradient for vertices:                     %g" % meanGradVert)
        print("  Volume-weighted mean function gradient for vertices:     %g" % wtMeanGradVert)
        print("  Minimum function gradient for vertices:                  %g" % minGradVert)
        print("  Maximum function gradient for vertices:                  %g" % maxGradVert)
        sys.stdout.flush()
    
        # Find value of gradient associated with fractionMin
        sortGradVert = numpy.argsort(gradVert)
        evalPoint = int((1.0 - self.gradFractionMin) * numVerts)
        valCutoff = gradVert[sortGradVert[evalPoint]]

        # Compute sizing function based on function gradient.
        maxScaleGrad = None
        minScaleGrad = None
        gradVertScale = None

        if (self.gradFunctionScaling == 'linear'):
            maxScaleGrad = valCutoff
            minScaleGrad = minGradVert
            gradVertScale = gradVert
        elif (self.gradFunctionScaling == 'log'):
            maxScaleGrad = numpy.log(valCutoff)
            minScaleGrad = numpy.log(minGradVert)
            gradVertScale = numpy.log(gradVert)
        elif (self.gradFunctionScaling == 'root2'):
            maxScaleGrad = numpy.sqrt(valCutoff)
            minScaleGrad = numpy.sqrt(minGradVert)
            gradVertScale = numpy.sqrt(gradVert)
        elif (self.gradFunctionScaling == 'root3'):
            maxScaleGrad = numpy.power(valCutoff, 1.0/3.0)
            minScaleGrad = numpy.power(minGradVert, 1.0/3.0)
            gradVertScale = numpy.power(gradVert, 1.0/3.0)
        elif (self.gradFunctionScaling == 'root8'):
            maxScaleGrad = numpy.power(valCutoff, 0.125)
            minScaleGrad = numpy.power(minGradVert, 0.125)
            gradVertScale = numpy.power(gradVert, 0.125)
        elif (self.gradFunctionScaling == 'root16'):
            maxScaleGrad = numpy.power(valCutoff, 0.0625)
            minScaleGrad = numpy.power(minGradVert, 0.0625)
            gradVertScale = numpy.power(gradVert, 0.0625)

        gradDiff = maxScaleGrad - minScaleGrad
        sizeDiff = self.gradMinSize - self.gradMaxSize
        slope = sizeDiff/gradDiff
        intercept = self.gradMinSize - slope * maxScaleGrad
        gradSizeFunc = slope * gradVertScale + intercept
        gradSizeFunc = numpy.maximum(gradSizeFunc, self.gradMinSize)
       
        # Match coordinates with Exodus mesh to get sizing function to apply.
        tree = scipy.spatial.cKDTree(coordsHdf, leafsize=10)
        (coordsMatch, coordIds) = tree.query(self.meshCoords)
        self.gradSizeFunc = gradSizeFunc[coordIds]
        meanGradSizeF = numpy.mean(self.gradSizeFunc)
        wtMeanGradSizeF = numpy.average(self.gradSizeFunc, weights=self.vertVolumes)
        minGradSizeF = numpy.amin(self.gradSizeFunc)
        maxGradSizeF = numpy.amax(self.gradSizeFunc)
        print("")
        print("  Mean gradient sizing function:                     %g" % meanGradSizeF)
        print("  Volume-weighted mean gradient sizing function:     %g" % wtMeanGradSizeF)
        print("  Minimum gradient sizing function:                  %g" % minGradSizeF)
        print("  Maximum gradient sizing function:                  %g" % maxGradSizeF)
        sys.stdout.flush()

        return

  
    def _calcGradSizeFuncTri(self):
        """
        Function to compute gradient sizing function for triangular cells.
        """

        print "Calculating gradient sizing function:"
        sys.stdout.flush()

        # Read HDF5 file.
        data = h5py.File(self.gradHdf5File, "r")
        coordsHdf = data['geometry/vertices'][:]
        numVerts = coordsHdf.shape[0]
        spaceDim = coordsHdf.shape[1]
        if (spaceDim != self.spaceDim):
            msg = "Spatial dimension of mesh must be equal to 3."
            raise ValueError(msg)
          
        cellConnect = numpy.array(data['topology/cells'][:], dtype=numpy.int64)
        numCells = cellConnect.shape[0]
        numVertsPerCell = cellConnect.shape[1]
        if (numVertsPerCell !=3):
            msg = "Only triangular cells may be used."
            raise ValueError(msg)

        fields = data['cell_fields']
        strain = fields['total_strain'][0,:,:]
        stress = fields['stress'][0,:,:]

        # Compute function for which to take gradient.
        if (self.gradFunctionType == 'strain_energy'):
            functionVals = self._tensorInnerProd(strain, stress)
        elif (self.gradFunctionType == 'sqrt_strain_energy'):
            functionVals = numpy.sqrt(self._tensorInnerProd(strain, stress))
        elif (self.gradFunctionType == 'strain_norm'):
            functionVals = numpy.sqrt(self._tensorInnerProd(strain, strain))
        elif (self.gradFunctionType == 'stress_norm'):
            functionVals = numpy.sqrt(self._tensorInnerProd(stress, stress))

        # Loop over cells and compute areas and function values at vertices.
        functionWtVert = numpy.zeros(numVerts, dtype=numpy.float64)
        areaVert = numpy.zeros(numVerts, dtype=numpy.float64)
        print("")
        print("  Computing cell areas and function values at vertices:")
        for cellNum in range(numCells):
            if (cellNum % self.printIncr == 0):
                print("    Working on cell # %d:" % cellNum)
                sys.stdout.flush()
            cellVerts = cellConnect[cellNum,:]
            cellCoords = coordsHdf[cellVerts,:].transpose()
            cellArea = self._triArea(cellCoords)/3.0
            areaVert[cellVerts] += cellArea
            functionWtVert[cellVerts] += cellArea * functionVals[cellNum]

        # Divide by vertex areas to get vertex function values.
        functionVert = functionWtVert/areaVert

        # Loop over cells and compute function gradient for each cell and vertex.
        gradCell = numpy.zeros(numCells, dtype=numpy.float64)
        gradWtVert = numpy.zeros(numVerts, dtype=numpy.float64)

        print("")
        print("  Computing function gradient for cells and vertices:")
        for cellNum in range(numCells):
            if (cellNum % self.printIncr == 0):
                print("    Working on cell # %d:" % cellNum)
                sys.stdout.flush()
            cellVerts = cellConnect[cellNum,:]
            refCoord = coordsHdf[cellVerts[0]]
            addCoords = coordsHdf[cellVerts[1:3]]
            diffCoords = addCoords - refCoord
            refFunction = functionVert[cellVerts[0]]
            addFunction = functionVert[cellVerts[1:3]]
            diffFunction = addFunction - refFunction
            gradient = numpy.linalg.solve(diffCoords, diffFunction)
            gradNorm = numpy.linalg.norm(gradient)
            gradCell[cellNum] = gradNorm
            gradWtVert[cellVerts] += areaVert[cellVerts] * gradCell[cellNum]

        gradVert = gradWtVert/areaVert

        # Print out gradient info for cells and vertices.
        meanGradCell = numpy.mean(gradCell)
        minGradCell = numpy.amin(gradCell)
        maxGradCell = numpy.amax(gradCell)
        print("")
        print("  Mean function gradient for cells:     %g" % meanGradCell)
        print("  Minimum function gradient for cells:  %g" % minGradCell)
        print("  Maximum function gradient for cells:  %g" % maxGradCell)
        meanGradVert = numpy.mean(gradVert)
        wtMeanGradVert = numpy.average(gradVert, weights=areaVert)
        minGradVert = numpy.amin(gradVert)
        maxGradVert = numpy.amax(gradVert)
        print("")
        print("  Mean function gradient for vertices:                   %g" % meanGradVert)
        print("  Area-weighted mean function gradient for vertices:     %g" % wtMeanGradVert)
        print("  Minimum function gradient for vertices:                %g" % minGradVert)
        print("  Maximum function gradient for vertices:                %g" % maxGradVert)
        sys.stdout.flush()
    
        # Find value of gradient associated with fractionMin
        sortGradVert = numpy.argsort(gradVert)
        evalPoint = int((1.0 - self.gradFractionMin) * numVerts)
        valCutoff = gradVert[sortGradVert[evalPoint]]

        # Compute sizing function based on function gradient.
        maxScaleGrad = None
        minScaleGrad = None
        gradVertScale = None

        if (self.gradFunctionScaling == 'linear'):
            maxScaleGrad = valCutoff
            minScaleGrad = minGradVert
            gradVertScale = gradVert
        elif (self.gradFunctionScaling == 'log'):
            maxScaleGrad = numpy.log(valCutoff)
            minScaleGrad = numpy.log(minGradVert)
            gradVertScale = numpy.log(gradVert)
        elif (self.gradFunctionScaling == 'root2'):
            maxScaleGrad = numpy.sqrt(valCutoff)
            minScaleGrad = numpy.sqrt(minGradVert)
            gradVertScale = numpy.sqrt(gradVert)
        elif (self.gradFunctionScaling == 'root3'):
            maxScaleGrad = numpy.power(valCutoff, 1.0/3.0)
            minScaleGrad = numpy.power(minGradVert, 1.0/3.0)
            gradVertScale = numpy.power(gradVert, 1.0/3.0)
        elif (self.gradFunctionScaling == 'root8'):
            maxScaleGrad = numpy.power(valCutoff, 0.125)
            minScaleGrad = numpy.power(minGradVert, 0.125)
            gradVertScale = numpy.power(gradVert, 0.125)
        elif (self.gradFunctionScaling == 'root16'):
            maxScaleGrad = numpy.power(valCutoff, 0.0625)
            minScaleGrad = numpy.power(minGradVert, 0.0625)
            gradVertScale = numpy.power(gradVert, 0.0625)

        gradDiff = maxScaleGrad - minScaleGrad
        sizeDiff = self.gradMinSize - self.gradMaxSize
        slope = sizeDiff/gradDiff
        intercept = self.gradMinSize - slope * maxScaleGrad
        gradSizeFunc = slope * gradVertScale + intercept
        gradSizeFunc = numpy.maximum(gradSizeFunc, self.gradMinSize)
       
        # Match coordinates with Exodus mesh to get sizing function to apply.
        tree = scipy.spatial.cKDTree(coordsHdf, leafsize=10)
        (coordsMatch, coordIds) = tree.query(self.meshCoords)
        self.gradSizeFunc = gradSizeFunc[coordIds]
        meanGradSizeF = numpy.mean(self.gradSizeFunc)
        wtMeanGradSizeF = numpy.average(gradSizeFunc, weights=areaVert)
        minGradSizeF = numpy.amin(self.gradSizeFunc)
        maxGradSizeF = numpy.amax(self.gradSizeFunc)
        print("")
        print("  Mean gradient sizing function:                   %g" % meanGradSizeF)
        print("  Area-weighted mean gradient sizing function:     %g" % wtMeanGradSizeF)
        print("  Minimum gradient sizing function:                %g" % minGradSizeF)
        print("  Maximum gradient sizing function:                %g" % maxGradSizeF)
        sys.stdout.flush()

        return


    def _getNodeSet(self, nodeSetName):
        """
        Function to get nodes in a nodeset.
        """

        nsSet = ""
        for nodeSet in range(self.numNodeSets):
            nsName = netCDF4.chartostring(self.exodus.variables['ns_names'][nodeSet])
            if (nsName == nodeSetName):
                nsNumber = nodeSet + 1
                nsSet = 'node_ns' + repr(nsNumber)
                break

        nodesInSet = self.exodus.variables[nsSet][:] - 1

        return nodesInSet
    
    
    def _getConstSizes(self, nsName, nsSize, nsSampleFreq):
        """
        Function to get nodeset coordinates and assign sizes for constant size values.
        """

        nodeSet = self._getNodeSet(nsName)
        coords = self.meshCoords[nodeSet, :]
        coordSample = coords[0::nsSampleFreq, :]
        coordSize = coordSample.shape[0]
        size = nsSize * numpy.ones(coordSize, dtype=numpy.float64)

        return (coordSample, size)


    def _getVarSizes(self, nsName, nsEdge1Name, nsEdge1Size, nsEdge2Name, nsEdge2Size, nsSampleFreq):
        """
        Function to compute linear variation in sizes along a surface.
        """

        # Get nodesets.
        ns = self._getNodeSet(nsName)
        nsEdge1 = self._getNodeSet(nsEdge1Name)
        nsEdge2 = self._getNodeSet(nsEdge2Name)

        # Get nodeset coordinates.
        surfCoords = self.meshCoords[ns,:]
        edge1Coords = self.meshCoords[nsEdge1,:]
        edge2Coords = self.meshCoords[nsEdge2,:]

        # Compute distances.
        dist1 = scipy.spatial.distance.cdist(surfCoords, edge1Coords)
        minDist1 = numpy.amin(dist1, axis=1)
        dist2 = scipy.spatial.distance.cdist(surfCoords, edge2Coords)
        minDist2 = numpy.amin(dist2, axis=1)

        # Compute distance ratios and associated sizes.
        totDist = minDist1 + minDist2
        r = minDist1/totDist
        sizes = nsEdge1Size * (1.0 - r) + r * nsEdge2Size

        # Resample.
        surfCoordsResamp = surfCoords[0::nsSampleFreq, :]
        sizesResamp = sizes[0::nsSampleFreq]

        return (surfCoordsResamp, sizesResamp)
    
    
    def _calcRbfSizeFunc(self):
        """
        Function to compute RBF sizing function.
        """

        #******** See about putting in point sizes as for MT code.
        # Maybe this code can be generalized to handle all cases.
        print("")
        print("Calculating RBF sizing function:")
        sys.stdout.flush()

        coordsX = self.meshCoords[:,0]
        coordsY = self.meshCoords[:,1]
        coordsZ = self.meshCoords[:,2]

        # Get nodeset info
        numConstSets = len(self.rbfConstNsNames)
        numVarSets = len(self.rbfVarNsNames)
        
        totalPoints = None
        totalSizes = None
        numConstPoints = 0
        numVarPoints = 0

        # Constant sizes.
        if (numConstSets != 0):
            for constSetNum in range(numConstSets):
                (points, sizes) = self._getConstSizes(self.rbfConstNsNames[constSetNum],
                                                      self.rbfConstNsSizes[constSetNum],
                                                      self.rbfConstNsSampfreq[constSetNum])
                if (constSetNum == 0):
                    totalPoints = points
                    totalSizes = sizes
                else:
                    totalPoints = numpy.append(totalPoints, points, axis=0)
                    totalSizes = numpy.append(totalSizes, sizes)

            numConstPoints = totalPoints.shape[0]
      
        # Variable sizes.
        if (numVarSets != 0):
            for varSetNum in range(numVarSets):
                (points, sizes) = self._getVarSizes(self.rbfVarNsNames[varSetNum], self.rbfVarNsEdge1Names[varSetNum],
                                                    self.rbfVarNsEdge1Sizes[varSetNum],
                                                    self.rbfVarNsEdge2Names[varSetNum],
                                                    self.rbfVarNsEdge2Sizes[varSetNum],
                                                    self.rbfVarNsSampleFreq[varSetNum])
                if (varSetNum == 0 and numConstSets == 0):
                    totalPoints = points
                    totalSizes = sizes
                else:
                    totalPoints = numpy.append(totalPoints, points, axis=0)
                    totalSizes = numpy.append(totalSizes, sizes)
            numVarPoints = totalPoints.shape[0] - numConstPoints
        
        xPoints = totalPoints[:,0]
        yPoints = totalPoints[:,1]
        zPoints = totalPoints[:,2]
        sizePoints = self.rbfSizeScale * totalSizes
        sizeMin = numpy.amin(sizePoints)
        sizeMax = numpy.amax(sizePoints)
        sizeDiff = sizeMax - sizeMin

        # Compute RBF function.
        print "  Setting up RBF interpolation:"
        rbfFunc = scipy.interpolate.Rbf(xPoints, yPoints, zPoints, sizePoints, function=self.rbfType,
                                        smooth=self.rbfSmoothing)

        print("  Computing RBF solution at mesh vertices:")
        vertsPerSlice = self.numMeshVerts/self.rbfNumMeshSlices
        sizeFunc = numpy.zeros(self.numMeshVerts, dtype=numpy.float64)
        for sliceNum in range(self.rbfNumMeshSlices):
            print("    Working on slice # %d:" % sliceNum)
            startInd = sliceNum * vertsPerSlice
            finishInd = min(startInd + vertsPerSlice, self.numMeshVerts)
            sizeFunc[startInd:finishInd] = rbfFunc(coordsX[startInd:finishInd],
                                                   coordsY[startInd:finishInd],
                                                   coordsZ[startInd:finishInd])
      
        # Rescale back to original min and max.
        sizeFuncMin = numpy.amin(sizeFunc)
        sizeFuncMax = numpy.amax(sizeFunc)
        minVals = numpy.where(sizeFunc < sizeMin)
        maxVals = numpy.where(sizeFunc > sizeMax)
        self.rbfSizeFunc = sizeFunc.copy()
        self.rbfSizeFunc[minVals] = sizeMin
        self.rbfSizeFunc[maxVals] = sizeMax

        # Print out some info on the sizing function.
        meanSizeF = numpy.mean(sizeFunc)
        wtMeanSizeF = numpy.average(sizeFunc, weights=self.vertVolumes)
        minSizeF = numpy.amin(sizeFunc)
        maxSizeF = numpy.amax(sizeFunc)
        print("")
        print("  Mean raw RBF sizing function:                     %g" % meanSizeF)
        print("  Volume-weighted mean raw RBF sizing function:     %g" % wtMeanSizeF)
        print("  Minimum raw RBF sizing function:                  %g" % minSizeF)
        print("  Maximum raw RBF sizing function:                  %g" % maxSizeF)
        print("")
        meanSizeFScaled = numpy.mean(self.rbfSizeFunc)
        wtMeanSizeFScaled = numpy.average(self.rbfSizeFunc, weights=self.vertVolumes)
        minSizeFScaled = numpy.amin(self.rbfSizeFunc)
        maxSizeFScaled = numpy.amax(self.rbfSizeFunc)
        print("  Mean scaled RBF sizing function:                     %g" % meanSizeFScaled)
        print("  Volume-weighted mean scaled RBF sizing function:     %g" % wtMeanSizeFScaled)
        print("  Minimum scaled RBF sizing function:                  %g" % minSizeFScaled)
        print("  Maximum scaled RBF sizing function:                  %g" % maxSizeFScaled)
        print("")

        return


    def _writeSizeFunc(self):
        """
        Function to add size function info to Exodus II file.
        """

        print ""
        print "Writing size function information:"
        sys.stdout.flush()

        meanTotF = numpy.mean(self.totSizeFunc)
        wtMeanTotF = numpy.average(self.totSizeFunc, weights=self.vertVolumes)
        minTotF = numpy.amin(self.totSizeFunc)
        maxTotF = numpy.amax(self.totSizeFunc)

        print("  Mean total sizing function:                     %g" % meanTotF)
        print("  Volume-weighted mean total sizing function:     %g" % wtMeanTotF)
        print("  Minimum total sizing function:                  %g" % minTotF)
        print("  Maximum total sizing function:                  %g" % maxTotF)
        print("")
        sys.stdout.flush()
        numNodVar = numpy.int32(4)
        if self.largeExodus:
            numNodVar = numpy.int64(4)

        # Add functions to database
        if not 'num_nod_var' in self.exodus.dimensions.keys():
            self.exodus.createDimension('num_nod_var', numNodVar)

            name_nod_var = self.exodus.createVariable('name_nod_var', 'S1', ('num_nod_var', 'len_string',))
            name_nod_var[0,:] = netCDF4.stringtoarr("cell_size", 33)
            name_nod_var[1,:] = netCDF4.stringtoarr("ellipse_cell_size", 33)
            name_nod_var[2,:] = netCDF4.stringtoarr("gradient_cell_size", 33)
            name_nod_var[3,:] = netCDF4.stringtoarr("rbf_cell_size", 33)
    
            vals_nod_var1 = self.exodus.createVariable('vals_nod_var1', numpy.float64, ('time_step', 'num_nodes',))
            vals_nod_var2 = self.exodus.createVariable('vals_nod_var2', numpy.float64, ('time_step', 'num_nodes',))
            vals_nod_var3 = self.exodus.createVariable('vals_nod_var3', numpy.float64, ('time_step', 'num_nodes',))
            vals_nod_var4 = self.exodus.createVariable('vals_nod_var4', numpy.float64, ('time_step', 'num_nodes',))
    
        zeros = numpy.zeros_like(self.totSizeFunc)
        time_whole = self.exodus.variables['time_whole']
        time_whole[0] = 0.0
        vals_nod_var1 = self.exodus.variables['vals_nod_var1']
        vals_nod_var2 = self.exodus.variables['vals_nod_var2']
        vals_nod_var3 = self.exodus.variables['vals_nod_var3']
        vals_nod_var4 = self.exodus.variables['vals_nod_var4']
        vals_nod_var1[0,:] = self.totSizeFunc.transpose()

        if (self.useEllSizeFunc):
            vals_nod_var2[0,:] = self.ellSizeFunc.transpose()
        else:
            vals_nod_var2[0,:] = zeros
        if (self.useGradSizeFunc):
            vals_nod_var3[0,:] = self.gradSizeFunc.transpose()
        else:
            vals_nod_var3[0,:] = zeros
        if (self.useRbfSizeFunc):
            vals_nod_var4[0,:] = self.rbfSizeFunc.transpose()
        else:
            vals_nod_var4[0,:] = zeros

        self.exodus.close()

        return


# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = CreateCellSize3D()
    app.run()

# End of file
