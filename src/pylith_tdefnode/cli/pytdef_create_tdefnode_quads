#!/usr/bin/env python
"""
This is a script to read a Tdefnode .nod file containing fault definitions and
create Cubit/Trelis journal files to generate the corresponding geometry.
Note that the .nod files from Defnode/Tdefnode are assumed to be WGS84 geographic
coordinates.
"""

# import pdb
import numpy as np
import math
from pyproj import Transformer
from fortranformat import FortranRecordReader
import platform

# If we are running Python 2, we will also assume PyLith 2.
PYTHON_MAJOR_VERSION = int(platform.python_version_tuple()[0])

if (PYTHON_MAJOR_VERSION == 2):
    from pyre.applications.Script import Script as Application
    from pyre.units import parser as unitparser
    from pyre.units.length import km
    from pyre.units.length import m
    from pyre.units.angle import degree
else:
    from pythia.pyre.applications.Script import Script as Application
    from pythia.pyre.units import parser as unitparser
    from pythia.pyre.units.length import km
    from pythia.pyre.units.length import m
    from pythia.pyre.units.angle import degree

class CreateTdefnodeQuads(Application):
    """
    This is a script to read a Tdefnode .nod file containing fault definitions and
    create Cubit/Trelis journal files to generate the corresponding geometry.
    """

    ## Python object for managing CreateTdefnodeQuads facilities and properties.
    ##
    ## \b Properties
    ## @li \b nod_file Defnode/Tdefnode .nod file to read.
    ## @li \b extend_file File containing the extension amount for each fault.
    ## @li \b file_type Defnode or Tdefnode input file.
    ## @li \b near_surf_eps Points this close will be projected to surface.
    ## @li \b output_projection Proj4 parameters defining output projection.
    ## @li \b output_rotation Rotation applied after projection.

    if (PYTHON_MAJOR_VERSION == 2):
        import pyre.inventory as inventory
    else:
        import pythia.pyre.inventory as inventory

    nodFile = inventory.str("nod_file", default="faults.nod")
    nodFile.meta['tip'] = "The name of the .nod file to read."
    
    extendFile = inventory.str("extend_file", default="extend.txt")
    extendFile.meta['tip'] = "File with the extension amount for each fault."

    fileType = inventory.str("file_type", default="tdefnode", validator = inventory.choice(["defnode", "tdefnode"]))
    fileType.meta['tip'] = "DEFNODE or TDEFNODE .nod file."

    nearSurfEps = inventory.dimensional("near_surf_eps", default=100.0*m)
    nearSurfEps.meta['tip'] = "Points this close will be projected to surface."

    outputProjection = inventory.str("output_projection",
                                     default="+proj=tmerc +lon_0=175.45 +lat_0=-40.825 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0")
    outputProjection.meta['tip'] = "Proj parameters defining output projection."

    outputRotation = inventory.dimensional("output_rotation", default=45.0*degree)
    outputRotation.meta['tip'] = "Rotation applied after projection."
                                    

    # PUBLIC METHODS /////////////////////////////////////////////////////

    def __init__(self, name="create_tdefnode_quads"):
        Application.__init__(self, name)
        self.numFaults = 0
        self.faultExtendsUpDict = {}
        self.faultExtendsDownDict = {}
        self.faultExtendsAsNegDict = {}
        self.faultExtendsAsPosDict = {}

        self.outProj = None
        self.WGS84 = "EPSG:4326"

        self.charEnd = 10

        self.newLine = "\n"
        self.separator = "# -----------------------------------------------------\n"
        self.comment = "#"

        return


    def main(self):
        # pdb.set_trace()
        self._readExtends()
        self._processFaults()

        return
                                    

    # PRIVATE METHODS /////////////////////////////////////////////////////


    def _configure(self):
        """
        Setup members using inventory.
        """
        Application._configure(self)
        rot = self.outputRotation.value
        cosR = math.cos(rot)
        sinR = math.sin(rot)
        self.rotMat = np.array([[ cosR, -sinR, 0.0],
                                [ sinR,  cosR, 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)

        if (self.fileType == "defnode"):
            nodFmt = "(a10, 3i3, 2(1x,a4), 1x, 3f9.3, 2f10.1, 4f8.1, f8.4," + \
                " 2f10.1, 6f8.1, 2f10.4, 1pe10.3)"
            self.nodFmt = FortranRecordReader(nodFmt)
        else:
            nodFmt = "(a10, 3i4, 2(1x,a4), 1x, 3f9.3, 2f10.3, 4f8.1, f8.4," + \
                " 8f8.1, 1x, 1pe10.3)"
            self.nodFmt = FortranRecordReader(nodFmt)

        self.transWGS84ToOutput = Transformer.from_crs(self.WGS84, self.outputProjection, always_xy=True)

        return
    

    def _readExtends(self):
        """
        Function to read amount of extension for each fault and create a dict.
        """
        f = open(self.extendFile, 'r')
        lines = f.readlines()
        faultNames = []
        faultUpExtends = []
        faultDownExtends = []
        faultAsNegExtends = []
        faultAsPosExtends = []
        uparser = unitparser()
        for lineNum in range(len(lines)):
            line = lines[lineNum]
            if (line.startswith('#') == False):
                faultNames.append(line.split()[0])
                extendUpAmount = uparser.parse(line.split()[1]).value
                extendDownAmount = uparser.parse(line.split()[2]).value
                extendAsNegAmount = uparser.parse(line.split()[3]).value
                extendAsPosAmount = uparser.parse(line.split()[4]).value
                faultUpExtends.append(extendUpAmount)
                faultDownExtends.append(extendDownAmount)
                faultAsNegExtends.append(extendAsNegAmount)
                faultAsPosExtends.append(extendAsPosAmount)

        self.faultExtendsUpDict = dict(zip(faultNames, faultUpExtends))
        self.faultExtendsDownDict = dict(zip(faultNames, faultDownExtends))
        self.faultExtendsAsNegDict = dict(zip(faultNames, faultAsNegExtends))
        self.faultExtendsAsPosDict = dict(zip(faultNames, faultAsPosExtends))

        return
      

    def _processFaults(self):
        """
        Function to loop over faults and get/write mesh info for each one.
        """

        print("Looping over faults:")

        # First find number of faults.
        f = open(self.nodFile, 'r')
        lines = f.readlines()
        numLines = len(lines)

        names = [lines[i][0:self.charEnd] for i in range(numLines)]
        uniqNames = list(set(names))
        self.numFaults = len(uniqNames)
        print("  Number of faults:  %d" % self.numFaults)

        # Loop over faults.
        for faultNum in range(self.numFaults):
            lowIndex = names.index(uniqNames[faultNum])
            highIndex = lowIndex + names.count(uniqNames[faultNum])
            self._processFault(lines[lowIndex:highIndex])
            
        return
  
    
    def _processFault(self, lines):
        """
        Function to process a single fault.
        """

        faultName = lines[0][0:self.charEnd].strip()
        print("  Working on fault:  %s" % faultName)

        numNodes = len(lines)

        xIndex = np.zeros(numNodes, dtype=np.int64)
        zIndex = np.zeros(numNodes, dtype=np.int64)
        lon = np.zeros(numNodes, dtype=np.float64)
        lat = np.zeros(numNodes, dtype=np.float64)
        elev = np.zeros(numNodes, dtype=np.float64)

        numAsNodes = 0
        numDdNodes = 0

        for lineNum in range(numNodes):
            vals = self.nodFmt.read(lines[lineNum])
            xIndex[lineNum] = vals[2]
            zIndex[lineNum] = vals[3]
            lon[lineNum] = vals[6]
            lat[lineNum] = vals[7]
            elev[lineNum] = -1000.0*vals[8]
            numAsNodes = max(numAsNodes, xIndex[lineNum])
            numDdNodes = max(numDdNodes, zIndex[lineNum])

        (x, y, z) = self.transWGS84ToOutput.transform(lon, lat, elev)
        coords = np.column_stack((x, y, z))
        coordsRot = np.dot(self.rotMat, coords.transpose()).transpose()
        (coordsExt, numAsNodesExt, numDdNodesExt) = self._createExtCoords(coordsRot, numAsNodes, numDdNodes,
                                                                          self.faultExtendsUpDict[faultName],
                                                                          self.faultExtendsDownDict[faultName],
                                                                          self.faultExtendsAsNegDict[faultName],
                                                                          self.faultExtendsAsPosDict[faultName])

        numNodesExt = numAsNodesExt*numDdNodesExt
        vertInds = np.arange(1, numNodesExt + 1, step=1, dtype=np.int32).reshape((numAsNodesExt, numDdNodesExt))

        # Write VTK file.
        self._writeVtk(faultName, coordsExt, numNodesExt, numAsNodesExt, numDdNodesExt)

        masterJournal = faultName + '_master.jou'
        vertexJournal = faultName + '_vertex.jou'
        surfJournal = faultName + '_surf.jou'

        # Write master journal file.
        self._writeMasterJournal(faultName, masterJournal, vertexJournal, surfJournal)

        # Write vertex definition journal file.
        self._writeVertJournal(vertexJournal, coordsExt, numAsNodesExt, numDdNodesExt)

        # Write surface definition journal file.
        self._writeSurfJournal(surfJournal, vertInds, numAsNodesExt, numDdNodesExt)

        return


    def _writeSurfJournal(self, surfJournal, vertInds, numAsNodes, numDdNodes):
        """
        Function to create Tdefnode quadrilaterals from vertices.
        """

        s = open(surfJournal, 'w')
        surfFmt = "create surface curve %d %d %d %d\n"
        curveFmt = "create curve vertex %d %d\n"
        cNum = 0
        curveInds = np.zeros((numAsNodes -1, numDdNodes - 1, 4), dtype=np.int32)

        # First create curves along strike.
        for asNode in range(numAsNodes - 1):
            cNum += 1
            for ddNode in range(numDdNodes - 1):
                v1 = vertInds[asNode, ddNode]
                v2 = vertInds[asNode + 1, ddNode]
                s.write(curveFmt % (v1, v2))
                if (ddNode == numDdNodes - 2):
                    v1 = vertInds[asNode, ddNode + 1]
                    v2 = vertInds[asNode + 1, ddNode + 1]
                    s.write(curveFmt % (v1, v2))
                curveInds[asNode, ddNode, 0] = cNum
                curveInds[asNode, ddNode, 2] = cNum + 1
                cNum += 1

        # Create curves down dip.
        for ddNode in range(numDdNodes - 1):
            cNum += 1
            for asNode in range(numAsNodes - 1):
                v1 = vertInds[asNode, ddNode]
                v2 = vertInds[asNode, ddNode + 1]
                s.write(curveFmt % (v1, v2))
                if (asNode == numAsNodes - 2):
                    v1 = vertInds[asNode + 1, ddNode]
                    v2 = vertInds[asNode + 1, ddNode + 1]
                    s.write(curveFmt % (v1, v2))
                curveInds[asNode, ddNode, 1] = cNum
                curveInds[asNode, ddNode, 3] = cNum + 1
                cNum += 1
        
        # Create surfaces with straight line curves.
        for asNode in range(numAsNodes - 1):
            for ddNode in range(numDdNodes - 1):
                c1 = curveInds[asNode, ddNode, 0]
                c2 = curveInds[asNode, ddNode, 1]
                c3 = curveInds[asNode, ddNode, 2]
                c4 = curveInds[asNode, ddNode, 3]
                s.write(surfFmt % (c1, c2, c3, c4))

        s.close()

        return
    
    
    def _writeVertJournal(self, vertexJournal, coords, numAsNodes, numDdNodes):
        """
        Function to write vertices defining Tdefnode quads.
        """

        v = open(vertexJournal, 'w')
        vertFmt = "create vertex %15.11e %15.11e %15.11e\n"

        for row in range(numAsNodes):
            for column in range(numDdNodes):
                v.write(vertFmt % (coords[row, column, 0], coords[row, column, 1], coords[row, column, 2]))


        v.close()

        return
  
    
    def _writeMasterJournal(self, faultName, masterJournal, vertexJournal, surfJournal):
        """
        Function to write the master journal file for geometry creation.
        """

        m = open(masterJournal, 'w')

        # Playback other parts.
        playCmd = \
          "reset\n" + \
          "# This is a simple Cubit journal file to create a volume\n" + \
          "# containing a set of quadrilateral surfaces from a set of\n" + \
          "# DEFNODE vertices.\n" + \
          self.comment + self.newLine + self.separator + \
          "# Create vertices, then quadrilaterals, then curves to define\n" + \
          "# a surrounding volume.\n" + \
          self.separator + \
          "playback " + "'" + vertexJournal + "'" + "\n" + \
          "playback " + "'" + surfJournal + "'" + "\n"

        m.write(playCmd)

        # Imprint and merge.
        comment1 = self.comment + self.newLine + self.separator + \
          "# Imprint and merge all surfaces.\n" + self.separator
        m.write(comment1)
        # imprintCmd = "imprint all\n" + \
          #              "merge all\n"
        mergeCmd = "merge all\n"
        m.write(mergeCmd)

        # Export surfaces.
        cubitFilename = faultName + "_cubit.cub"
        comment2 = self.comment + self.newLine + self.separator + \
          "# Export DEFNODE volumes.\n" + self.separator
        m.write(comment2)
        exportCubit = "export Cubit '" + cubitFilename + "' surface all overwrite\n"
        m.write(exportCubit)
        m.close()
    
        return
    
    
    def _createExtCoords(self, coords, numAsNodes, numDdNodes, upExt, downExt, asNegExt, asPosExt):
        """
        Function to create extended coordinates array.
        """

        zMax = np.amax(coords[:,2])
        numAsNodesExt = numAsNodes + 2
        numDdNodesExt = numDdNodes + 2
        topNode = 1
        coordsReshape = coords.reshape((numAsNodes, numDdNodes, 3), order='F')
        coordsExt = np.zeros((numAsNodesExt, numDdNodesExt, 3))

        # Extend updip, if possible.
        top = coordsReshape[:,0,:]
        atSurface = np.allclose(top[:,2], 0.0, atol=self.nearSurfEps.value)

        if (atSurface):
            numDdNodesExt = numDdNodes + 1
            topNode = 0
            coordsExt = np.zeros((numAsNodesExt, numDdNodesExt, 3))
            coordsExt[1:-1,topNode:-1,:] = coordsReshape
            coordsExt[:,topNode,2] = self.nearSurfEps.value
        else:
            coordsExt[1:-1,topNode:-1,:] = coordsReshape
            diff = coordsReshape[:,0,:] - coordsReshape[:,1,:]
            diffNorm = np.linalg.norm(diff, axis=1)
            diffVec = diff/diffNorm.reshape((numAsNodes, 1))
            testTop = coordsReshape[:,topNode,:] + upExt*diffVec
            coordsExt[1:-1,0,:] = testTop
            aboveSurf = np.any(testTop[:,2] >= 0.0)
            if (aboveSurf):
                coordsExt[1:-1,0,2] = self.nearSurfEps.value

        # Extend downdip.
        diff = coordsReshape[:,-1,:] - coordsReshape[:,-2,:]
        diffNorm = np.linalg.norm(diff, axis=1)
        diffVec = diff/diffNorm.reshape((numAsNodes, 1))
        bottom = coordsReshape[:,-1,:] + downExt*diffVec
        coordsExt[1:-1,-1,:] = bottom

        # Extend left.
        diff = coordsExt[1,:,:] - coordsExt[2,:,:]
        diffNorm = np.linalg.norm(diff, axis=1)
        diffVec = diff/diffNorm.reshape((numDdNodesExt, 1))
        left = coordsExt[1,:,:] + asNegExt*diffVec
        coordsExt[0,:,:] = left

        # Extend right.
        diff = coordsExt[-2,:,:] - coordsExt[-3,:,:]
        diffNorm = np.linalg.norm(diff, axis=1)
        diffVec = diff/diffNorm.reshape((numDdNodesExt, 1))
        right = coordsExt[-2,:,:] + asPosExt*diffVec
        coordsExt[-1,:,:] = right

        return (coordsExt, numAsNodesExt, numDdNodesExt)
      
    
    def _writeVtk(self, faultName, coords, numNodes, numAsNodes, numDdNodes):
        """
        Function to write coordinates and cell connectivity info.
        """

        fileName = faultName + '.vtk'
        v = open(fileName, 'w')
        # VTK headers.
        headerTop = "# vtk DataFile Version 2.0\n" + \
            "Extended fault geometry from DEFNODE\n" + \
            "ASCII\n" + \
            "DATASET UNSTRUCTURED_GRID\n"
        headerTotal = headerTop + \
            "POINTS " + repr(numNodes) + " double\n"
        v.write(headerTotal)

        coordsOut = coords.reshape((numNodes, 3))
        np.savetxt(v, coordsOut)
    
        # Determine connectivity.
        numCells = (numAsNodes - 1) * (numDdNodes - 1)
        numCellEntries = 5 * numCells
        headConnect = "CELLS %d %d\n" % (numCells, numCellEntries)
        connect = np.zeros((numCells, 5), dtype=np.int32)
        connect[:,0] = 4
        headCellType = "CELL_TYPES %d\n" % numCells
        cellType = 9 * np.ones(numCells, dtype=np.int32)
        intFmt = "%d"

        cellNum = 0
        for dipCell in range(numDdNodes - 1):
            for strikeCell in range(numAsNodes - 1):
                v1 = dipCell + strikeCell * numDdNodes
                v2 = v1 + 1
                v3 = v2 + numDdNodes
                v4 = v1 + numDdNodes
                connect[cellNum,1] = v1
                connect[cellNum,2] = v2
                connect[cellNum,3] = v3
                connect[cellNum,4] = v4
                cellNum += 1

        # Write connectivity info.
        v.write(headConnect)
        np.savetxt(v, connect, fmt=intFmt)
        v.write(headCellType)
        np.savetxt(v,cellType, fmt=intFmt)

        v.close()

        return
    
    
# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = CreateTdefnodeQuads()
    app.run()

# End of file
