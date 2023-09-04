#!/usr/bin/env nemesis
#
# ----------------------------------------------------------------------
#
# Charles A. Williams, GNS Science
#
# ----------------------------------------------------------------------
#

## @file get_tdef_moment.py

## @brief Python application to read TDefnode fault info and compute associated shear modulus.

import pdb
import platform

# For now, if we are running Python 2, we will also assume PyLith 2.
PYTHON_MAJOR_VERSION = int(platform.python_version_tuple()[0])

if (PYTHON_MAJOR_VERSION == 2):
    from pyre.applications.Script import Script as Application
else:
    from pythia.pyre.applications.Script import Script as Application

import numpy as np


class getTdefMoment(Application):
    """
    Python application to read TDefnode fault info and compute associated shear modulus.
    """
  
    ## \b Properties
    ## @li \b tdef_flt_info_file Name of TDefnode fault attribute file.
    ## @li \b fault_normal_dist Distance to project on either side of fault to get properties.
    ## @li \b tdef_flt_info_output_file Name of modified fault attribute file to create.

    ## \b Facilities
    ## @li \b db_velocity Spatial database for seismic velocity.
    ## @li \b coordsys_tdefnode Coordinate system associated with TDefnode output.
    ## @li \b coordsys_local Coordinate system used for local coordinate calculations.

    if (PYTHON_MAJOR_VERSION == 2):
        import pyre.inventory as inventory
    else:
        import pythia.pyre.inventory as inventory


    tdefFltInfoFile = inventory.str("tdef_flt_info_file", default="mod_flt_001_atr.gmt")
    tdefFltInfoFile.meta['tip'] = "Name of TDefnode fault attribute file."

    faultNormalDist = inventory.float("fault_normal_dist", default=100.0)
    faultNormalDist.meta['tip'] = "Name of modified TDefnode fault attribute file to create."

    tdefFltInfoOutputFile = inventory.str("tdef_flt_info_output_file", default="mod_flt_001_shearmod_atr.gmt")
    tdefFltInfoOutputFile.meta['tip'] = "Name of modified TDefnode fault attribute file to create."

    from spatialdata.spatialdb.SimpleDB import SimpleDB
    from spatialdata.spatialdb.SimpleGridDB import SimpleGridDB
    dbVelocity = inventory.facility("db_velocity", family="spatial_database", factory=SimpleGridDB)
    dbVelocity.meta['tip'] = "Spatial database for seismic velocity."

    from spatialdata.geocoords.CSGeo import CSGeo
    coordsysTDefnode = inventory.facility("coordsys_tdefnode", family="coordsys", factory=CSGeo)
    coordsysTDefnode.meta['tip'] = "Coordinate system associated with TDefnode output."

    coordsysLocal = inventory.facility("coordsys_local", family="coordsys", factory=CSGeo)
    coordsysLocal.meta['tip'] = "Coordinate system used for local coordinate calculations."


    # PUBLIC METHODS /////////////////////////////////////////////////////

    def __init__(self, name="get_tdef_moment"):
        Application.__init__(self, name)

        return


    def main(self):
        # pdb.set_trace()
        self._getMoment()
        return


    # PRIVATE METHODS ////////////////////////////////////////////////////

    def _configure(self):
        """
        Setup members using inventory.
        """
        Application._configure(self)

        return


    def _getMoment(self):
        """
        Open output file from TDefnode, compute effective shear modulus on either side of fault,
        and output modified TDefnode file.
        """

        i = open(self.tdefFltInfoFile, 'r')
        w = open(self.tdefFltInfoOutputFile, 'w')

        lines = i.readlines()
        numLines = len(lines)
        numPatches = numLines // 5
        checkLines = numLines % 5
        if (checkLines != 0):
            msg = "Number of lines is not divisible by 5."
            raise ValueError(msg)

        lineStart = 0
        lineEnd = lineStart + 4

        # Loop over patches to get sample coordinates.
        coordsPos = np.zeros((numPatches, 3), dtype=np.float64)
        coordsNeg = np.zeros((numPatches, 3), dtype=np.float64)
        for patchNum in range(numPatches):
            lineStart = patchNum*5
            lineEnd = lineStart + 5
            (coordsPos[patchNum,:], coordsNeg[patchNum,:]) = self._getSampleCoords(lines[lineStart:lineEnd])

        # Query spatial database to get Vs and density.
        dataPos = np.zeros((numPatches, 2), dtype=np.float64)
        dataNeg = np.zeros((numPatches, 2), dtype=np.float64)
        err = np.zeros((numPatches,), dtype=np.int32)
        csLocal = self.coordsysLocal

        db = self.dbVelocity
        db.open()
        db.setQueryValues(["density", "vs"])
        db.multiquery(dataPos, err, coordsPos, csLocal)
        db.multiquery(dataNeg, err, coordsNeg, csLocal)
        densityPos = dataPos[:,0]
        densityNeg = dataNeg[:,0]
        vsPos = dataPos[:,1]
        vsNeg = dataNeg[:,1]
        db.close()

        # Compute effective shear modulus.
        shearPos = densityPos*vsPos*vsPos
        shearNeg = densityNeg*vsNeg*vsNeg
        effShearModulus = 2.0*shearPos*shearNeg/(shearPos + shearNeg)
        
        # Loop over patches.
        for patchNum in range(numPatches):
            lineStart = patchNum*5
            lineEnd = lineStart + 5
            lineHead = lines[lineStart].strip()
            effShearModulusStr = f'{effShearModulus[patchNum]:.6g}'
            lineHeadMod = lineHead + '  ' + effShearModulusStr + '\n'
            w.write(lineHeadMod)
            for lineNum in range(lineStart + 1, lineEnd):
                w.write(lines[lineNum])

        i.close()
        w.close()

        return
                
        
    def _getSampleCoords(self, lines):
        """
        Get sample coordinates for subsegment by sampling +/- normal distances to the fault.
        """
        # Get header line and make empty arrays.
        from spatialdata.geocoords.Converter import convert
        headline = lines[0]
        lons = np.zeros(4, dtype=np.float64)
        lats = np.zeros(4, dtype=np.float64)
        depths = np.zeros(4, dtype=np.float64)

        # Get geographic coordinates of corners.
        for vertNum in range(4):
            lineSplit = lines[vertNum + 1].split()
            lons[vertNum] = float(lineSplit[0])
            lats[vertNum] = float(lineSplit[1])
            depths[vertNum] = 1000.0*float(lineSplit[2])

        # Convert coordinates to local Cartesian system and compute normals.
        coords = np.column_stack((lons, lats, depths))
        csLocal = self.coordsysLocal
        csTDefnode = self.coordsysTDefnode
        convert(coords, csLocal, csTDefnode)
        v1a = coords[1,:] - coords[0,:]
        v2a = coords[3,:] - coords[0,:]
        v1b = coords[3,:] - coords[2,:]
        v2b = coords[1,:] - coords[2,:]
        normalA = np.cross(v1a, v2a)
        normalA /= np.linalg.norm(normalA)
        normalB = np.cross(v1b, v2b)
        normalB /= np.linalg.norm(normalB)
        normal = 0.5*(normalA + normalB)
        normal /= np.linalg.norm(normal)

        # Get cell center and distances on either side of patch.
        cellCenter = np.mean(coords, axis=0)
        coordsPos = (cellCenter + normal*self.faultNormalDist).reshape(1,3)
        coordsNeg = (cellCenter - normal*self.faultNormalDist).reshape(1,3)

        return(coordsPos, coordsNeg)
                
        
# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = getTdefMoment()
    app.run()

# End of file
