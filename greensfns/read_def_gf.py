#!/usr/bin/env python
"""
This is a script to read Green's function info from DEFNODE and create a
VTK file. This version forms quadrilateral cells.
"""

import pdb
import os
import glob
import platform

# For now, if we are running Python 2, we will also assume PyLith 2.
PYTHON_MAJOR_VERSION = int(platform.python_version_tuple()[0])

if (PYTHON_MAJOR_VERSION == 2):
    from pathlib2 import Path
    from pyre.applications.Script import Script as Application
    import pyre.units.unitparser
    from pyre.units.length import km
    from pyre.units.length import mm
else:
    from pathlib import Path
    from pythia.pyre.applications.Script import Script as Application
    import pythia.pyre.units.unitparser
    from pythia.pyre.units.length import km
    from pythia.pyre.units.length import mm

import numpy
import scipy.spatial
from pyproj import Proj
from pyproj import transform
from fortranformat import FortranRecordReader

class ReadDefGf(Application):
    """
    This is a script to read Green's function info from DEFNODE/TDEFNODE
    and create VTK output.
    """

    ## Python object for managing ReadDefGf facilities and properties.
    ##
    ## \b Properties
    ## @li \b defnode_gf_dir Directory containing Defnode GF.
    ## @li \b gf_type Use DEFNODE or TDEFNODE Green's functions.
    ## @li \b defnode_fault_num Fault number for which to read GF.
    ## @li \b vtk_impulse_root Root filename for VTK impulse output.
    ## @li \b vtk_response_root Root filename for VTK response output.
    ## @li \b vtk_response_prefix Prefix given to response output names.
    ## @li \b output_projection Proj4 parameters defining output projection.

    if (PYTHON_MAJOR_VERSION == 2):
        import pyre.inventory as inventory
    else:
        import pythia.pyre.inventory as inventory

    defnodeGfDir = inventory.str("defnode_gf_dir", default="m10b")
    defnodeGfDir.meta['tip'] = "Directory containing Defnode Green's functions."

    gfType = inventory.str("gf_type", default="defnode", validator=inventory.choice(["defnode", "tdefnode"]))
    gfType.meta['tip'] = "Use DEFNODE or TDEFNODE Green's functions."

    defnodeFaultNum = inventory.int("defnode_fault_num", default=1)
    defnodeFaultNum.meta['tip'] = "Fault number for which to read GF."

    vtkImpulseRoot = inventory.str("vtk_impulse_root", default="greensfns_impulse")
    vtkImpulseRoot.meta['tip'] = "Root filename for VTK impulse output."

    vtkResponseRoot = inventory.str("vtk_response_root", default="greensfns_response")
    vtkResponseRoot.meta['tip'] = "Root filename for VTK response output."

    vtkResponsePrefix = inventory.str("vtk_response_prefix", default="tdef_")
    vtkResponsePrefix.meta['tip'] = "Prefix given to response output names."

    outputProjection = inventory.str("output_projection",
                                          default="+proj=tmerc +lon_0=175.45 +lat_0=-40.825 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0")
    outputProjection.meta['tip'] = "Proj4 parameters defining output projection."
                                    
    
    # PUBLIC METHODS /////////////////////////////////////////////////////

    def __init__(self, name="read_def_gf"):
        Application.__init__(self, name)
        self.numImpulses = 0
        self.numGpsFiles = 0
        self.numInsarFiles = 0
        self.numUpFiles = 0
        self.numGpsResponses = 0
        self.numInsarResponses = 0
        self.numUpResponses = 0
        self.numAsNodes = 0
        self.numDdNodes = 0
        self.GpsFiles = []
        self.InsarFiles = []
        self.UpFiles = []
        self.faultCoords = None
        self.responseGpsCoords = None
        self.responseInsarCoords = None
        self.responseUpCoords = None
        self.defGpsSitesUse = []
        self.defInsarSitesUse = []
        self.defUpSitesUse = []

        self.responseE = None
        self.responseN = None

        self.outProj = None
        WGS84 = "+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0"
        self.projWGS84 = Proj(WGS84)

        return


    def main(self):
        # pdb.set_trace()
        self.outProj = Proj(self.outputProjection)
        self._getFileLists()
        self._readDefnodeGf()
        self._writeImpulses()

        return
                                    

    # PRIVATE METHODS /////////////////////////////////////////////////////

    def _configure(self):
        """
        Setup members using inventory.
        """
        Application._configure(self)

        gfHeadFmt = "(a1,3i4,i5,3f12.5,2f8.2,2i5,a5, 1x, a12, d14.7, f10.4)"
        gfGpsFmt = "(a1, 2f10.4, 1x, (4d20.13))"
        gfInsarFmt = "(a1, 2f10.4, 1x, (6d20.13))"
        gfUpFmt = "(a1,2f10.4,(2d20.13), 1x, a8)"

        if (self.gfType == 'tdefnode'):
            gfGpsFmt = "(a1, 2f10.4, 1x, (6d20.13))"

        self.gfHeadFmt = FortranRecordReader(gfHeadFmt)
        self.gfGpsFmt = FortranRecordReader(gfGpsFmt)
        self.gfInsarFmt = FortranRecordReader(gfInsarFmt)
        self.gfUpFmt = FortranRecordReader(gfUpFmt)

        pathName = os.path.dirname(self.vtkImpulseRoot)
        outputDir = self._checkDir(pathName)
        pathName = os.path.dirname(self.vtkResponseRoot)
        outputDir = self._checkDir(pathName)

        return


    def _checkDir(self, subDir):
        """
        Function to see if directory exists and create it if necessary.
        """
        if os.path.isabs(subDir):
            newDir = subDir
        else:
            newDir = os.path.join(os.getcwd(), subDir)

        testDir = os.path.isdir(newDir)
        testFile = os.path.isfile(newDir)

        if (testDir == False):
            if (testFile == True):
                msg = "Subdirectory exists as a file."
                raise ValueError(msg)
            else:
                os.makedirs(newDir)

        return newDir


    def _getHeaderInfo(self, gfType):
        """
        Function to extract info from Defnode/TDefnode file headers.
        """
        files = self.GpsFiles
        if (gfType == 'insar'):
            files = self.InsarFiles
        if (gfType == 'up'):
            files = self.UpFiles

        maxNumSites = 0
        maxSitesGf = 0
        for fileNum in range(len(files)):
            f = open(files[fileNum], 'r')
            line = f.readline()
            (pref, kf, ix, iz, numSites, lon, lat, depth, gfX, gfW, ish, nlay, defVers, date, xMom, gpsNear) = self.gfHeadFmt.read(line)
            self.numAsNodes = max(self.numAsNodes, ix)
            self.numDdNodes = max(self.numDdNodes, iz)
            if (numSites > maxNumSites):
                maxNumSites = numSites
                maxSitesGf = fileNum
            f.close()

        return (maxNumSites, maxSitesGf)


    def _getFileLists(self):
        """
        Function to get lists of GPS, InSAR, and Up files.
        """

        print "Getting lists of Green's functions:"

        totalGfPath = os.path.normpath(os.path.join(os.getcwd(), self.defnodeGfDir))
        faultString = 'gf' + repr(self.defnodeFaultNum).rjust(3, '0')
        searchGps = os.path.join(totalGfPath, faultString + '*g')
        self.GpsFiles = glob.glob(searchGps)
        self.GpsFiles.sort()
        self.numGpsFiles = len(self.GpsFiles)
        if (self.numGpsFiles != 0):
            self.numImpulses = self.numGpsFiles
            (self.numGpsResponses, self.gpsReprSite) = self._getHeaderInfo('gps')

        if (self.gfType == 'tdefnode'):
            searchInsar = os.path.join(totalGfPath, faultString + '*i')
            self.InsarFiles = glob.glob(searchInsar)
            self.InsarFiles.sort()
            self.numInsarFiles = len(self.InsarFiles)
            if (self.numInsarFiles != 0 and self.numGpsFiles != 0):
                if (self.numInsarFiles != self.numImpulses):
                    msg = "Number of GF mismatch for GPS and InSAR files."
                    raise ValueError(msg)
            if (self.numInsarFiles != 0):
                self.numImpulses = self.numInsarFiles
                (self.numInsarResponses, self.insarReprSite) = self._getHeaderInfo('insar')


        if (self.gfType == 'defnode'):
            searchUp = os.path.join(totalGfPath, faultString + '*u')
            self.UpFiles = glob.glob(searchUp)
            self.UpFiles.sort()
            self.numUpFiles = len(self.UpFiles)
            if (self.numUpFiles != 0 and self.numGpsFiles != 0):
                if (self.numUpFiles != self.numImpulses):
                    msg = "Number of GF mismatch for GPS and Up files."
                    raise ValueError(msg)

            if (self.numUpFiles != 0):
                self.numImpulses = self.numUpFiles
            if (self.numGpsFiles == 0):
                (self.numUpResponses, self.upReprSite) = self._getHeaderInfo('up')

        if (self.numGpsFiles != 0):
            self.gpsResponseCoordsGeog = self._getSiteCoords(self.GpsFiles[self.gpsReprSite], 'gps')
            self.gpsResponseCoords = numpy.zeros((self.numGpsResponses, 3), dtype=numpy.float64)
            (self.gpsResponseCoords[:,0], self.gpsResponseCoords[:,1]) = transform(
                self.projWGS84, self.outProj, self.gpsResponseCoordsGeog[:,0], self.gpsResponseCoordsGeog[:,1])
            self.gpsResponseTree = scipy.spatial.cKDTree(self.gpsResponseCoordsGeog)
        if (self.numInsarFiles != 0):
            self.insarResponseCoordsGeog = self._getSiteCoords(self.InsarFiles[self.insarReprSite], 'insar')
            self.insarResponseCoords = numpy.zeros((self.numInsarResponses, 3), dtype=numpy.float64)
            (self.insarResponseCoords[:,0], self.insarResponseCoords[:,1]) = transform(
                self.projWGS84, self.outProj, self.insarResponseCoordsGeog[:,0], self.insarResponseCoordsGeog[:,1])
            self.insarResponseTree = scipy.spatial.cKDTree(self.insarResponseCoordsGeog)
        if (self.numUpFiles != 0):
            self.upResponseCoordsGeog = self._getSiteCoords(self.UpFiles[self.upReprSite], 'up')
            self.upResponseCoords = numpy.zeros((self.numUpResponses, 3), dtype=numpy.float64)
            (self.upResponseCoords[:,0], self.upResponseCoords[:,1]) = transform(
                self.projWGS84, self.outProj, self.upResponseCoordsGeog[:,0], self.upResponseCoordsGeog[:,1])
            self.upResponseTree = scipy.spatial.cKDTree(self.upResponseCoordsGeog)
        return
  
    
    def _getSiteCoords(self, fileName, dataType):
        """
        Get site coordinates from the given file.
        """
        lineFmt = self.gfGpsFmt
        if (dataType == 'gps'):
            lineFmt = self.gfGpsFmt
        elif (dataType == 'insar'):
            lineFmt = self.gfInsarFmt
        elif (dataType == 'up'):
            lineFmt = self.gfUpFmt

        f = open(fileName, 'r')
        lines = f.readlines()
        numLines = len(lines)
        numSites = numLines - 1
        coords = numpy.zeros((numSites, 2), dtype=numpy.float64)
        siteNum = 0

        for lineNum in range(1, numLines):
            vars = lineFmt.read(lines[lineNum])
            coords[siteNum,0] = vars[1]
            coords[siteNum,1] = vars[2]
            siteNum += 1

        f.close()

        return coords


    def _matchCoords(self, lon, lat, refCoordTree):
        """
        Match lon/lat coords from file to reference coordinates.
        """

        eps = 0.001
        coordsFind = numpy.column_stack((lon, lat))
        coordInds = refCoordTree.query_ball_point(coordsFind, eps)
        checkLen = numpy.array([len(i)>1 for i in coordInds])
        if (numpy.any(checkLen)):
            msg = 'More than one matching site coordinate found.'
            raise ValueError(msg)
        matchInds = numpy.array([i[0] for i in coordInds])

        return matchInds


    def _readUpFile(self, fileNum):
        """
        Function to read a Defnode Up GF file.
        """

        outFile = self.vtkResponseRoot + '_up_r' + repr(fileNum).rjust(4, '0') + ".vtk"
        o = open(outFile, 'w')
        i = open(self.UpFiles[fileNum], 'r')
        lines = i.readlines()

        (pref, kf, ix, iz, numSites, lon, lat, depth, gfX, gfW, ish, nlay, defVers, date, xMom, gpsNear) = self.gfHeadFmt.read(lines[0])
        self.numAsNodes = max(self.numAsNodes, ix)
        self.numDdNodes = max(self.numDdNodes, iz)
        elev = -1000.0 * depth
        coords = [lon, lat, elev]

        lonR = numpy.zeros(numSites, dtype=numpy.float64)
        latR = numpy.zeros(numSites, dtype=numpy.float64)
        elevR = numpy.zeros(numSites, dtype=numpy.float64)
        zRespE = numpy.zeros(numSites, dtype=numpy.float64)
        zRespN = numpy.zeros(numSites, dtype=numpy.float64)

        # Loop over lines in file.
        for lineNum in range(1, len(lines)):
            respNum = lineNum - 1
            (pref, lonR[respNum], latR[respNum], zRespE[respNum], zRespN[respNum], stn) = self.gfUpFmt.read(lines[lineNum])

        i.close()

        coordInds = self._matchCoords(lonR, latr, self.upResponseTree)
        zRespEOut = numpy.zeros(self.numUpResponses, dtype=numpy.float64)
        zRespNOut = numpy.zeros(self.numUpResponses, dtype=numpy.float64)
        zRespEOut[coordInds] = zRespE
        zRespNOut[coordInds] = zRespN
      
        vtkHead = "# vtk DataFile Version 2.0\n" + \
            "Response for Defnode Up GF\n" + \
            "ASCII\n" + \
            "DATASET POLYDATA\n" + \
            "POINTS %d double\n" % self.numUpResponses

        o.write(vtkHead)
        numpy.savetxt(o, self.upResponseCoords)

        responseNameE = self.vtkResponsePrefix + "z_response_e"
        responseNameN = self.vtkResponsePrefix + "z_response_n"
        vtkHead2 = "POINT_DATA %d\n" % numSites
        vtkHead2a = "SCALARS %s float 1\nLOOKUP_TABLE default\n" % responseNameE
        o.write(vtkHead2)
        o.write(vtkHead2a)
        numpy.savetxt(o, zRespEOut)

        vtkHead3 = "SCALARS %s float 1\nLOOKUP_TABLE default\n" % responseNameN
        o.write(vtkHead3)
        numpy.savetxt(o, zRespNOut)

        o.close()
        
        return coords
  
    
    def _readGfFile(self, fileNum, gfType):
        """
        Function to read a Defnode/TDefnode GF file.
        """

        responseNameE = self.vtkResponsePrefix + "response_e"
        responseNameN = self.vtkResponsePrefix + "response_n"
        responseNameT = self.vtkResponsePrefix + "response_total"
        outFile = self.vtkResponseRoot + '_gps_r' + repr(fileNum).rjust(4, '0') + ".vtk"
        vtkInfoLine = "Response for TDefnode GPS GF\n"
        files = self.GpsFiles
        refCoords = self.gpsResponseCoordsGeog
        refCoordTree = self.gpsResponseTree
        outCoords = self.gpsResponseCoords
        numResponses = self.numGpsResponses
        if (gfType == 'insar'):
            outFile = self.vtkResponseRoot + '_insar_r' + repr(fileNum).rjust(4, '0') + ".vtk"
            vtkInfoLine = "Response for TDefnode InSAR GF\n"
            files = self.InsarFiles
            refCoords = self.insarResponseCoordsGeog
            refCoordTree = self.insarResponseTree
            outCoords = self.insarResponseCoords
            numResponses = self.numInsarResponses

        if (self.gfType == 'defnode'):
            vtkInfoLine = "Response for Defnode GPS GF\n"

        o = open(outFile, 'w')

        i = open(files[fileNum], 'r')
        lines = i.readlines()

        (pref, kf, ix, iz, numSites, lon, lat, depth, gfX, gfW, ish, nlay, defVers, date, xMom, gpsNear) = self.gfHeadFmt.read(lines[0])
        self.numAsNodes = max(self.numAsNodes, ix)
        self.numDdNodes = max(self.numDdNodes, iz)
        elev = -1000.0 * depth
        coords = [lon, lat, elev]

        lonR = numpy.zeros(numSites, dtype=numpy.float64)
        latR = numpy.zeros(numSites, dtype=numpy.float64)
        elevR = numpy.zeros(numSites, dtype=numpy.float64)
        xRespE = numpy.zeros(numSites, dtype=numpy.float64)
        yRespE = numpy.zeros(numSites, dtype=numpy.float64)
        zRespE = numpy.zeros(numSites, dtype=numpy.float64)
        xRespN = numpy.zeros(numSites, dtype=numpy.float64)
        yRespN = numpy.zeros(numSites, dtype=numpy.float64)
        zRespN = numpy.zeros(numSites, dtype=numpy.float64)

        # Loop over lines in file.

        for lineNum in range(1, len(lines)):
            respNum = lineNum - 1
            if (self.gfType == 'defnode'):
                (pref, lonR[respNum], latR[respNum], xRespE[respNum], yRespE[respNum],
                 xRespN[respNum], yRespN[respNum]) = self.gfGpsFmt.read(lines[lineNum])
            else:
                (pref, lonR[respNum], latR[respNum], xRespE[respNum], xRespN[respNum], yRespE[respNum], yRespN[respNum],
                 zRespE[respNum], zRespN[respNum]) = self.gfGpsFmt.read(lines[lineNum])

        i.close()
        coordInds = self._matchCoords(lonR, latR, refCoordTree)
        xRespEOut = numpy.zeros(numResponses, dtype=numpy.float64)
        yRespEOut = numpy.zeros(numResponses, dtype=numpy.float64)
        zRespEOut = numpy.zeros(numResponses, dtype=numpy.float64)
        xRespNOut = numpy.zeros(numResponses, dtype=numpy.float64)
        yRespNOut = numpy.zeros(numResponses, dtype=numpy.float64)
        zRespNOut = numpy.zeros(numResponses, dtype=numpy.float64)
        xRespEOut[coordInds] = xRespE
        yRespEOut[coordInds] = yRespE
        zRespEOut[coordInds] = zRespE
        xRespNOut[coordInds] = xRespN
        yRespNOut[coordInds] = yRespN
        zRespNOut[coordInds] = zRespN
      
        vtkHead = "# vtk DataFile Version 2.0\n" + \
            vtkInfoLine + \
            "ASCII\n" + \
            "DATASET POLYDATA\n" + \
            "POINTS %d double\n" % numResponses

        o.write(vtkHead)
        numpy.savetxt(o, outCoords)

        vtkHead2 = "POINT_DATA %d\n" % numResponses
        vtkHead2a = "VECTORS %s double\n" % responseNameE
        o.write(vtkHead2)
        o.write(vtkHead2a)
        responseE = numpy.column_stack((xRespEOut, yRespEOut, zRespEOut))
        numpy.savetxt(o, responseE)

        vtkHead3 = "VECTORS %s double\n" % responseNameN
        o.write(vtkHead3)
        responseN = numpy.column_stack((xRespNOut, yRespNOut, zRespNOut))
        numpy.savetxt(o, responseN)

        vtkHead4 = "VECTORS %s double\n" % responseNameT
        o.write(vtkHead4)
        totResponse = responseE + responseN
        numpy.savetxt(o, totResponse)

        o.close()

        return coords


    def _readDefnodeGf(self):
        """
        Function to read Defnode Green's function files.
        """

        print "Reading Defnode Green's function files:"

        impulseGeogCoords = []
        printIncr = 10
    
        # Loop over GPS files.
        for fileNum in range(self.numGpsFiles):
            if (fileNum % printIncr == 0):
                print "  Reading GPS file number:  %d" % fileNum
            gpsCoords = self._readGfFile(fileNum, 'gps')
            impulseGeogCoords.append(gpsCoords)

        if (self.numGpsFiles != 0):
            impulseGeogArray = numpy.array(impulseGeogCoords)
            self.faultCoords = numpy.zeros((self.numImpulses, 3), dtype=numpy.float64)
            lon = impulseGeogArray[:,0]
            lat = impulseGeogArray[:,1]
            elev = impulseGeogArray[:,2]
            (self.faultCoords[:,0], self.faultCoords[:,1], self.faultCoords[:,2]) = \
              transform(self.projWGS84, self.outProj, lon, lat, elev)
    
        # Loop over InSAR files.
        impulseGeogCoords2 = []
        for fileNum in range(self.numInsarFiles):
            if (fileNum % printIncr == 0):
                print "  Reading InSAR file number:  %d" % fileNum
            insarCoords = self._readGfFile(fileNum, 'insar')
            impulseGeogCoords2.append(insarCoords)

        if (self.numGpsFiles == 0 and self.numInsarFiles != 0):
            impulseGeogArray = numpy.array(impulseGeogCoords2)
            self.faultCoords = numpy.zeros((self.numImpulses, 3), dtype=numpy.float64)
            lon = impulseGeogArray[:,0]
            lat = impulseGeogArray[:,1]
            elev = impulseGeogArray[:,2]
            (self.faultCoords[:,0], self.faultCoords[:,1], self.faultCoords[:,2]) = transform(self.projWGS84, self.outProj, lon, lat, elev)

        if (self.gfType == 'defnode'):
            impulseGeogCoords3 = []
    
            # Loop over Up files.
            for fileNum in range(self.numUpFiles):
                if (fileNum % printIncr == 0):
                    print "  Reading Uplift file number:  %d" % fileNum
                upCoords = self._readUpFile(fileNum)
                impulseGeogCoords3.append(upCoords)

            if (self.numGpsFiles == 0):
                impulseGeogArray = numpy.array(impulseGeogCoords3)
                self.faultCoords = numpy.zeros((self.numImpulses, 3), dtype=numpy.float64)
                lon = impulseGeogArray[:,0]
                lat = impulseGeogArray[:,1]
                elev = impulseGeogArray[:,2]
                (self.faultCoords[:,0], self.faultCoords[:,1], self.faultCoords[:,2]) = transform(self.projWGS84, self.outProj, lon, lat, elev)

        return
      
    
    def _writeImpulses(self):
        """
        Function to write impulse information.
        """

        print "Writing impulse VTK files:"

        # VTK headers.
        headerTop = "# vtk DataFile Version 2.0\n" + \
            "Greens function impulse information from DEFNODE/TDEFNODE\n" + \
            "ASCII\n" + \
            "DATASET UNSTRUCTURED_GRID\n"
        headerTotal = headerTop + "POINTS " + repr(self.numImpulses) + " double\n"
        slip = numpy.zeros(self.numImpulses, dtype=numpy.float64)
        slipHead = "POINT_DATA %d\n" % self.numImpulses
        slipHeadTot = slipHead + "SCALARS fault_slip double 1\n" + "LOOKUP_TABLE default\n"

        # Determine connectivity.
        numCells = (self.numAsNodes - 1) * (self.numDdNodes - 1)
        numCellEntries = 5 * numCells
        headConnect = "CELLS %d %d\n" % (numCells, numCellEntries)
        connect = numpy.zeros((numCells, 5), dtype=numpy.int32)
        connect[:,0] = 4
        headCellType = "CELL_TYPES %d\n" % numCells
        cellType = 9 * numpy.ones(numCells, dtype=numpy.int32)
        intFmt = "%d"

        cellNum = 0
        for strikeCell in range(self.numAsNodes - 1):
            for dipCell in range(self.numDdNodes - 1):
                v1 = dipCell + strikeCell * self.numDdNodes
                v2 = v1 + 1
                v3 = v2 + self.numDdNodes
                v4 = v3 - 1
                connect[cellNum,1] = v1
                connect[cellNum,2] = v2
                connect[cellNum,3] = v3
                connect[cellNum,4] = v4
                cellNum += 1
    

        # Slip array.
        headSlip = "POINT_DATA " + repr(self.numImpulses) + "\n" + \
            "SCALARS fault_slip double 1\n" + \
            "LOOKUP_TABLE default\n"
        slip = numpy.zeros(self.numImpulses, dtype=numpy.float64)

        printIncr = 10

        # Loop over impulses, writing a separate file for each one.
        for impulseNum in range(self.numImpulses):
            if (impulseNum % printIncr == 0):
                print "  Writing impulse file number:  %d" % impulseNum
            slip[impulseNum] = 1.0
            if (impulseNum > 0):
                slip[impulseNum - 1] = 0.0
            fileName = self.vtkImpulseRoot + '_i' + repr(impulseNum).rjust(4, '0') + ".vtk"
            o = open(fileName, 'w')

            o.write(headerTotal)
            numpy.savetxt(o, self.faultCoords)
            o.write(headConnect)
            numpy.savetxt(o, connect, fmt=intFmt)
            o.write(headCellType)
            numpy.savetxt(o,cellType, fmt=intFmt)
            o.write(headSlip)
            numpy.savetxt(o, slip)
            o.close()

        return

# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = ReadDefGf()
    app.run()

# End of file
