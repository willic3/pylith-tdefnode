#!/usr/bin/env python
"""
This is a script to read Green's function info from DEFNODE and create a
VTK file. This version forms quadrilateral cells.
"""

import pdb
import os
import glob
import numpy
from pyproj import Proj
from pyproj import transform
from fortranformat import FortranRecordReader
import h5py
from pylith.meshio.Xdmf import Xdmf

from pyre.applications.Script import Script as Application

class ReadDefGf(Application):
    """
    This is a script to read Green's function info from DEFNODE/TDEFNODE
    and create HDF5/XDMF output.
    """

    import pyre.inventory
    ## Python object for managing ReadDefGf facilities and properties.
    ##
    ## \b Properties
    ## @li \b defnode_gf_dir Directory containing Defnode GF.
    ## @li \b gf_type Use DEFNODE or TDEFNODE Green's functions.
    ## @li \b defnode_fault_num Fault number for which to read GF.
    ## @li \b vtk_impulse_root Root filename for VTK impulse output.
    ## @li \b vtk_response_root Root filename for VTK response output.
    ## @li \b output_projection Proj4 parameters defining output projection.

    defnodeGfDir = pyre.inventory.str("defnode_gf_dir", default="m10b")
    defnodeGfDir.meta['tip'] = "Directory containing Defnode Green's functions."

    gfType = pyre.inventory.str("gf_type", default="defnode", validator=pyre.inventory.choice(["defnode", "tdefnode"]))
    gfType.meta['tip'] = "Use DEFNODE or TDEFNODE Green's functions."

    defnodeFaultNum = pyre.inventory.int("defnode_fault_num", default=1)
    defnodeFaultNum.meta['tip'] = "Fault number for which to read GF."

    vtkImpulseRoot = pyre.inventory.str("vtk_impulse_root", default="greensfns_impulse")
    vtkImpulseRoot.meta['tip'] = "Root filename for VTK impulse output."

    vtkResponseRoot = pyre.inventory.str("vtk_response_root", default="greensfns_response")
    vtkResponseRoot.meta['tip'] = "Root filename for VTK response output."

    outputProjection = pyre.inventory.str("output_projection",
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

        return


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
            (maxNumGpsSites, maxSitesGf) = self._getHeaderInfo('gps')

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
            if (self.numGpsFiles == 0):
                (maxNumGpsSites, maxSitesGf) = self._getHeaderInfo('insar')


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
                (maxNumGpsSites, maxSitesGf) = self._getHeaderInfo('up')

        return
  
    
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
      
        (x, y, z) = transform(self.projWGS84, self.outProj, lonR, latR, elevR)
        coordsR = numpy.column_stack((x, y, z))

        vtkHead = "# vtk DataFile Version 2.0\n" + \
            "Response for Defnode Up GF\n" + \
            "ASCII\n" + \
            "DATASET POLYDATA\n" + \
            "POINTS %d double\n" % numSites

        o.write(vtkHead)
        numpy.savetxt(o, coordsR)

        vtkHead2 = "POINT_DATA %d\n" % numSites
        vtkHead2a = "SCALARS z_response_e float 1\n" + \
            "LOOKUP_TABLE default\n"
        o.write(vtkHead2)
        o.write(vtkHead2a)
        numpy.savetxt(o, zRespE)

        vtkHead3 = "SCALARS z_response_n float 1\n" + \
            "LOOKUP_TABLE default\n"
        o.write(vtkHead3)
        numpy.savetxt(o, zRespN)

        o.close()
        
        return coords
  
    
    def _readGfFile(self, fileNum, gfType):
        """
        Function to read a Defnode/TDefnode GF file.
        """

        outFile = self.vtkResponseRoot + '_gps_r' + repr(fileNum).rjust(4, '0') + ".vtk"
        vtkInfoLine = "Response for TDefnode GPS GF\n"
        files = self.GpsFiles
        if (gfType == 'insar'):
            outFile = self.vtkResponseRoot + '_insar_r' + repr(fileNum).rjust(4, '0') + ".vtk"
            vtkInfoLine = "Response for TDefnode InSAR GF\n"
            files = self.InsarFiles

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
      
        (x, y, z) = transform(self.projWGS84, self.outProj, lonR, latR, elevR)
        coordsR = numpy.column_stack((x, y, z))

        vtkHead = "# vtk DataFile Version 2.0\n" + \
            vtkInfoLine + \
            "ASCII\n" + \
            "DATASET POLYDATA\n" + \
            "POINTS %d double\n" % numSites

        o.write(vtkHead)
        numpy.savetxt(o, coordsR)

        vtkHead2 = "POINT_DATA %d\n" % numSites
        vtkHead2a = "VECTORS response_e double\n"
        o.write(vtkHead2)
        o.write(vtkHead2a)
        responseE = numpy.column_stack((xRespE, yRespE, zRespE))
        numpy.savetxt(o, responseE)

        vtkHead3 = "VECTORS response_n double\n"
        o.write(vtkHead3)
        responseN = numpy.column_stack((xRespN, yRespN, zRespN))
        numpy.savetxt(o, responseN)

        vtkHead4 = "VECTORS total_response double\n"
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
