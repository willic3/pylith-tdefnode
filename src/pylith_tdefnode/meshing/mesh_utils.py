#!/usr/bin/env python
"""
Meshing utilities to be used when creating a PyLith mesh compatible with TDefnode.
"""

import pdb
import numpy as np
import math
from fortranformat import FortranRecordReader

newLine = "\n"
separator = "# -----------------------------------------------------\n"
comment = "#"

def writeCubitSurfJournal(surfJournal, vertInds, numAsNodes, numDdNodes):
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
    
    
def writeCubitVertJournal(vertexJournal, coords, numAsNodes, numDdNodes):
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
  
    
def writeCubitMasterJournal(faultName, masterJournal, vertexJournal, surfJournal):
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
        comment + newLine + separator + \
        "# Create vertices, then quadrilaterals, then curves to define\n" + \
        "# a surrounding volume.\n" + \
        separator + \
        "playback " + "'" + vertexJournal + "'" + "\n" + \
        "playback " + "'" + surfJournal + "'" + "\n"

    m.write(playCmd)

    # Imprint and merge.
    comment1 = comment + newLine + separator + \
        "# Merge all surfaces.\n" + separator
    m.write(comment1)
    mergeCmd = "merge all\n"
    m.write(mergeCmd)

    # Export surfaces.
    acisFilename = faultName + "_acis.sat"
    cubitFilename = faultName + "_cubit.cub"
    comment2 = comment + newLine + separator + \
        "# Export DEFNODE volumes.\n" + separator
    m.write(comment2)
    exportAcis = "export Acis '" + acisFilename + "' surface all overwrite\n"
    exportCubit = "export Cubit '" + cubitFilename + "' surface all overwrite\n"
    m.write(exportAcis)
    m.write(exportCubit)
    m.close()
    
    return
    
    
def createExtCoords(coords, numAsNodes, numDdNodes, upExt, downExt, asNegExt, asPosExt):
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
        testTop = coordsReshape[:,topNode,:] + upExt * diffVec
        coordsExt[1:-1,0,:] = testTop
        aboveSurf = np.any(testTop[:,2] >= 0.0)
        if (aboveSurf):
            coordsExt[1:-1,0,2] = self.nearSurfEps.value

    # Extend downdip.
    diff = coordsReshape[:,-1,:] - coordsReshape[:,-2,:]
    diffNorm = np.linalg.norm(diff, axis=1)
    diffVec = diff/diffNorm.reshape((numAsNodes, 1))
    bottom = coordsReshape[:,-1,:] + downExt * diffVec
    coordsExt[1:-1,-1,:] = bottom

    # Extend left.
    diff = coordsExt[1,:,:] - coordsExt[2,:,:]
    diffNorm = np.linalg.norm(diff, axis=1)
    diffVec = diff/diffNorm.reshape((numDdNodesExt, 1))
    left = coordsExt[1,:,:] + asNegExt * diffVec
    coordsExt[0,:,:] = left

    # Extend right.
    diff = coordsExt[-2,:,:] - coordsExt[-3,:,:]
    diffNorm = np.linalg.norm(diff, axis=1)
    diffVec = diff/diffNorm.reshape((numDdNodesExt, 1))
    right = coordsExt[-2,:,:] + asPosExt * diffVec
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
