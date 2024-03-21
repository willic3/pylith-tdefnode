"""
Package with functions to read and write Defnode/TDefnode files.
"""

import math
import os
import re
import sys
import glob
import platform

import numpy as np
from fortranformat import FortranRecordReader
from fortranformat import FortranRecordWriter
from pyproj import Transformer
import scipy.spatial
import scipy.spatial.distance
import shapely.geometry

geogCoordEpsilon = 1.0e-5
defUpGfFmt = '(A1, 2f10.4, (2d20.13), 1x, a8)'
defGGfFmt = '(A1, 2f10.4, 1x, (4d20.13))'
tdefGfFmt = '(A1, 2f10.4, 1x, (6d20.13))'
    
def readDefGf(self, gfFile, gfType, coordsGeog):
    """
    Function to read DEFNODE Green's function file, excluding header info.
    """
    coordsGeogCurrent = None
    if (coordsGeog is not None):
        coordsGeogCurrent = coordsGeog.copy()
    f = open(gfFile, 'r')
    fLines = f.readlines()
    numSites = len(fLines) - 1
    gfFortran = defGGfFmt
    if (gfType == 'u'):
        gfFortran = defUpGfFmt
    gfFmt = FortranRecordReader(gfFortran)
    gfCoordsGeog = np.zeros((numSites, 2), dtype=np.float64)
    sites = []
    pointNum = 0

    for lineNum in range(1, numSites + 1):
        if (gfType == 'g'):
            (type, gfCoordsGeog[pointNum,0], gfCoordsGeog[pointNum,1], gf1, gf2, gf3, gf4) = gfFmt.read(fLines[lineNum])
        else:
            (type, gfCoordsGeog[pointNum,0], gfCoordsGeog[pointNum,1], gf1, gf2, site) = gfFmt.read(fLines[lineNum])
            sites.append(site)
        if (coordsGeogCurrent is not None):
            diff = coordsGeogCurrent - gfCoordsGeog[pointNum,:]
            norm = np.linalg.norm(diff, axis=1)
            indMin = np.argmin(norm)
            if (norm[indMin] > epsilon):
                coordsGeogCurrent = np.append(coordsGeogCurrent, gfCoordsGeog[pointNum,:].reshape(1,2), axis=0)
        else:
            coordsGeogCurrent = gfCoordsGeog[pointNum,:].reshape(1,2)
        pointNum += 1

        (x, y) = transform(self.projWGS84, self.projPylith, coordsGeogCurrent[:,0], coordsGeogCurrent[:,1])
        coordsCartCurrent = np.column_stack((x, y))
        numTotalSites = coordsGeogCurrent.shape[0]

        if (gfType == 'g'):
            return (numTotalSites, coordsGeogCurrent, coordsCartCurrent)
        else:
            return (numTotalSites, coordsGeogCurrent, coordsCartCurrent, sites)

    
    def _readTDefGf(self, gfFile, coordsGeog):
        """
        Function to read TDEFNODE Green's function file.
        """
        epsilon = 1.0e-5
        coordsGeogCurrent = None
        if (coordsGeog is not None):
            coordsGeogCurrent = coordsGeog.copy()
        f = open(gfFile, 'r')
        fLines = f.readlines()
        numSites = len(fLines) - 1
        gfFortran = '(A1, 2f10.4, 1x, (6d20.13))'
        gfFmt = FortranRecordReader(gfFortran)
        gfCoordsGeog = np.zeros((numSites, 2), dtype=np.float64)
        pointNum = 0

        for lineNum in range(1, numSites + 1):
            (type, gfCoordsGeog[pointNum,0], gfCoordsGeog[pointNum,1], gf1, gf2, gf3, gf4, gf5, gf6) = gfFmt.read(fLines[lineNum])
            if (coordsGeogCurrent is not None):
                diff = coordsGeogCurrent - gfCoordsGeog[pointNum,:]
                norm = np.linalg.norm(diff, axis=1)
                indMin = np.argmin(norm)
                if (norm[indMin] > epsilon):
                    coordsGeogCurrent = np.append(coordsGeogCurrent, gfCoordsGeog[pointNum,:].reshape(1,2), axis=0)
            else:
                coordsGeogCurrent = gfCoordsGeog[pointNum,:].reshape(1,2)
            pointNum += 1

        (x, y) = transform(self.projWGS84, self.projPylith, coordsGeogCurrent[:,0], coordsGeogCurrent[:,1])
        coordsCartCurrent = np.column_stack((x, y))
        numTotalSites = coordsGeogCurrent.shape[0]

        return (numTotalSites, coordsGeogCurrent, coordsCartCurrent)

class Py2Def(Application):
    """
    Python application to combine PyLith-generated Green's functions to produce GF
    for DEFNODE or TDEFNODE.
    """
  
    ## \b Properties
    ## @li \b ll_impulse_file Filename for left-lateral HDF5 impulse file.
    ## @li \b ud_impulse_file Filename for updip HDF5 impulse file.
    ## @li \b nm_impulse_file Filename for normal slip HDF5 impulse file.
    ## @li \b ll_response_file Filename for left-lateral HDF5 response file.
    ## @li \b ud_response_file Filename for updip HDF5 response file.
    ## @li \b nm_response_file Filename for normal slip HDF5 response file.
    ## @li \b gf_type Use DEFNODE or TDEFNODE Green's functions.
    ## @li \b unmatched_site_option Exit with error or create zero values for sites without Green's functions.
    ## @li \b fault_projection_plane Plane into which to project to determine whether points lie in DEFNODE quads.
    ## @li \b defnode_gf_dir Directory containing DEFNODE/TDEFNODE Green's functions.
    ## @li \b defnode_fault_num DEFNODE/TDEFNODE fault number.
    ## @li \b defnode_fault_slip_type Use shear only or 3D (requires fault-normal Green's functions).
    ## @li \b mesh_coordsys Projection information for mesh coordinates.
    ## @li \b gf_output_directory Directory to put DEFNODE/TDEFNODE GF files.
    ## @li \b impulse_info_file Output VTK file with impulse information.
    ## @li \b response_info_root Root filename for VTK response information files.
    ## @li \b gf_scale Scaling factor to apply to Green's functions.
    ## @li \b site_match_epsilon Epsilon value for matching sites.
    ## @li \b node_match_epsilon Epsilon value for matching fault nodes.
    ## @li \b bilinear_coeff_epsilon Epsilon value for bilinear interpolation coefficients.
    """
    For now, assume I don't need to separately scale each GF type.
    ## @li \b ud_scale Scaling factor to apply to updip Green's functions.
    ## @li \b nm_scale Scaling factor to apply to normal slip Green's functions.
    """

    if (PYTHON_MAJOR_VERSION == 2):
        import pyre.inventory as inventory
    else:
        import pythia.pyre.inventory as inventory

    llImpulseFile = inventory.str("ll_impulse_file", default="gf_ll_fault.h5")
    llImpulseFile.meta['tip'] = "Filename of left-lateral impulse HDF5 file."

    udImpulseFile = inventory.str("ud_impulse_file", default="gf_ud_fault.h5")
    udImpulseFile.meta['tip'] = "Filename of updip impulse HDF5 file."

    nmImpulseFile = inventory.str("nm_impulse_file", default="gf_nm_fault.h5")
    nmImpulseFile.meta['tip'] = "Filename of normal slip impulse HDF5 file."

    llResponseFile = inventory.str("ll_response_file", default="gf_ll_points.h5")
    llResponseFile.meta['tip'] = "Filename of left-lateral response HDF5 file."

    udResponseFile = inventory.str("ud_response_file", default="gf_ud_points.h5")
    udResponseFile.meta['tip'] = "Filename of updip response HDF5 file."

    nmResponseFile = inventory.str("nm_response_file", default="gf_nm_points.h5")
    nmResponseFile.meta['tip'] = "Filename of normal slip response HDF5 file."

    gfType = inventory.str("gf_type", default="defnode", validator=inventory.choice(["defnode", "tdefnode"]))
    gfType.meta['tip'] = "Use DEFNODE or TDEFNODE Green's functions."

    unmatchedSiteOption = inventory.str("unmatched_site_option", default="zero", validator=inventory.choice(["zero", "error"]))
    unmatchedSiteOption.meta['tip'] = "Exit with error or create zero values for sites without Green's functions."

    faultProjectionPlane = inventory.str("fault_projection_plane", default="xy_plane",
                                         validator=inventory.choice(["xy_plane", "xz_plane", "yz_plane", "best_fit_plane",
                                                                     "defnode_endpoints"]))
    faultProjectionPlane.meta['tip'] = "Plane into which to project to determine whether points lie in DEFNODE quads."

    defnodeGfDir = inventory.str("defnode_gf_dir", default="defnode_gf")
    defnodeGfDir.meta['tip'] = "Directory containing DEFNODE/TDEFNODE Green's functions."

    defnodeFaultNum = inventory.int("defnode_fault_num", default=1)
    defnodeFaultNum.meta['tip'] = "DEFNODE/TDEFNODE fault number."

    defnodeFaultSlipType = inventory.str("defnode_fault_slip_type", default="shear", validator=inventory.choice(["shear", "3d"]))
    defnodeFaultSlipType.meta['tip'] = "Use shear only or 3D (requires fault-normal Green's functions)."

    meshCoordsys = inventory.str("mesh_coordsys",
                                 default="+proj=tmerc +lon_0=175.45 +lat_0=-40.825 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0")
    meshCoordsys.meta['tip'] = "Projection information for mesh coordinates."

    gfOutputDirectory = inventory.str("gf_output_directory", default="greensfns")
    gfOutputDirectory.meta['tip'] = "Directory to put DEFNODE/TDEFNODE GF files."

    impulseInfoFile = inventory.str("impulse_info_file", default="impulse_info.vtk")
    impulseInfoFile.meta['tip'] = "Output VTK file with impulse information."

    responseInfoRoot = inventory.str("response_info_root", default="response_info")
    responseInfoRoot.meta['tip'] = "Root filename for response information."

    gfScale = inventory.float("gf_scale", default=1.0)
    gfScale.meta['tip'] = "Scaling factor to apply to Green's functions."

    siteMatchEpsilon = inventory.float("site_match_epsilon", default=100.0)
    siteMatchEpsilon.meta['tip'] = "Epsilon value for matching sites (meters)."

    nodeMatchEpsilon = inventory.float("node_match_epsilon", default=100.0)
    nodeMatchEpsilon.meta['tip'] = "Epsilon value for matching fault nodes (meters)."

    bilinearCoeffEpsilon = inventory.float("bilinear_coeff_epsilon", default=0.02)
    bilinearCoeffEpsilon.meta['tip'] = "Epsilon value for bilinear interpolation coefficients."


    # PUBLIC METHODS /////////////////////////////////////////////////////

    def __init__(self, name="py2def"):
        Application.__init__(self, name)

        self.faultCoords = None
        self.faultCoordsLocal = None
        self.responseCoords = None
        self.numPatchesShared = None
        self.totalArea = None
        self.connectedPatches = None
        self.connectedPatchArea = None
        self.faultConnect = None
        self.patchArea = None
        self.faultSlip = None
        self.numFaultVerts = 0
        self.numFaultCells = 0
        self.numImpulses = 0
        self.numResponses = 0
        self.impulseVerts = []
        self.pyEastGPSGf = None
        self.pyNorthGPSGf = None
        self.pyEastUpGf = None
        self.pyNorthUpGf = None
        self.pyEastInsarGf = None
        self.pyNorthInsarGf = None
        self.strikeDeg = None
        self.dipDeg = None
        self.rakeDegE = None
        self.rakeDegN = None
        self.rakeDeg = None
        self.llContrib = None
        self.udContrib = None
        self.nmContrib = None
        self.llContribE = None
        self.udContribE = None
        self.nmContribE = None
        self.llContribN = None
        self.udContribN = None
        self.nmContribN = None

        self.defNodeCoords = None
        self.defNodeCoordsLocal = None
        self.defCellConnect = None
        self.numDefNodes = 0
        self.numDefCells = 0
        self.useGps = True
        self.useInsar = True
        self.useUp = True

        self.gpsCoordsGeog = None
        self.gpsCoordsCart = None
        self.insarCoordsGeog = None
        self.insarCoordsCart = None
        self.upCoordsGeog = None
        self.upCoordsCart = None
        self.upSites = []
        self.gpsIndices = []
        self.insarIndices = []
        self.upIndices = []
        self.gpsHeaders = []
        self.insarHeaders = []
        self.upHeaders = []
        self.defGfG = None
        self.defGfI = None
        self.defGfU = None
        self.numDefNodes = 0
        self.numAsNodes = 0
        self.numDdNodes = 0
        self.numGpsSites = 0
        self.numInsarSites = 0
        self.numUpSites = 0

        self.unmatchedGPSSites = None
        self.unmatchedGPSSiteCoords = None

        self.unmatchedInsarSites = None
        self.unmatchedInsarSiteCoords = None

        self.unmatchedUpSites = None
        self.unmatchedUpSiteCoords = None

        WGS84 = "+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0"
        self.projWGS84 = Proj(WGS84)
        self.projPylith = None

        self.quadWeight = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

        gfHeadFmt = "(a1,3i4,i5,3f12.5,2f8.2,2i5,a5, 1x, a12, d14.7, f10.4)"
        self.gfHeadFmtR = FortranRecordReader(gfHeadFmt)
        self.gfHeadFmtW = FortranRecordWriter(gfHeadFmt)

        return


    def main(self):
        # import pdb
        # pdb.set_trace()
        self.projPylith = Proj(self.meshCoordsys)
        print("Working on fault number %d:" % self.defnodeFaultNum)
        self._readDefnode()
        if (self.gfType == "defnode"):
            self.useInsar = False
            self._readPyGFDefnode()
        else:
            self.useUp = False
            self._readPyGFTDefnode()
        self._getAreas()
        self._integrateGF()
        if (self.gfType == "defnode"):
            self._writeDefnodeGF()
        else:
            self._writeTDefnodeGF()
        self._writeImpulseInfo()
        self._writeResponseInfo()
        if (self.unmatchedSiteOption == 'zero'):
            self._writeUnmatchedSites()
        return


    # PRIVATE METHODS ////////////////////////////////////////////////////

    def _configure(self):
        """
        Setup members using inventory.
        """
        Application._configure(self)

        return
      

    def _writeImpulseInfo(self):
        """
        Function to write impulse information to a VTK file.
        """

        print("  Writing impulse information:")
        sys.stdout.flush()

        outputDir = Path(self.impulseInfoFile).parents[0]
        outputDir.mkdir(parents=True, exist_ok=True)
        vtkHead = "# vtk DataFile Version 2.0\n" + \
                  "Impulse information for PyLith to Defnode conversion.\n" + \
                  "ASCII\n" + \
                  "DATASET POLYDATA\n" + \
                  "POINTS %d double\n" % self.numFaultVerts
        v = open(self.impulseInfoFile, 'w')
        v.write(vtkHead)
        np.savetxt(v, self.faultCoords)

        sharedHead1 = "POINT_DATA %d\n" % self.numFaultVerts
        sharedHead = sharedHead1 + "SCALARS num_shared_patches int 1\n" + \
                     "LOOKUP_TABLE default\n"
        v.write(sharedHead)
        np.savetxt(v, self.numPatchesShared, fmt="%d")

        slipHead = "SCALARS total_slip double 1\n" + \
                   "LOOKUP_TABLE default\n"
        v.write(slipHead)
        np.savetxt(v, self.faultSlip)

        impulses = -1 * np.ones(self.numFaultVerts, dtype=np.int64)
        impulses[self.impulseVerts] = 1
        impulseHead = "SCALARS impulse_applied int 1\n" + \
                      "LOOKUP_TABLE default\n"
        v.write(impulseHead)
        np.savetxt(v, impulses, fmt="%d")

        vertNums = np.arange(0, self.numFaultVerts, dtype=np.int64)
        vertHead = "SCALARS vertex_num int 1\n" + \
                   "LOOKUP_TABLE default\n"
        v.write(vertHead)
        np.savetxt(v, vertNums, fmt="%d")

        llHeadE = "SCALARS ll_contrib_east double 1\n" + \
                  "LOOKUP_TABLE default\n"
        v.write(llHeadE)
        np.savetxt(v, self.llContribE, fmt="%g")

        llHeadN = "SCALARS ll_contrib_north double 1\n" + \
                  "LOOKUP_TABLE default\n"
        v.write(llHeadN)
        np.savetxt(v, self.llContribN, fmt="%g")

        llHead = "SCALARS ll_contrib double 1\n" + \
                 "LOOKUP_TABLE default\n"
        v.write(llHead)
        np.savetxt(v, self.llContrib, fmt="%g")

        udHeadE = "SCALARS ud_contrib_east double 1\n" + \
                  "LOOKUP_TABLE default\n"
        v.write(udHeadE)
        np.savetxt(v, self.udContribE, fmt="%g")

        udHeadN = "SCALARS ud_contrib_north double 1\n" + \
                  "LOOKUP_TABLE default\n"
        v.write(udHeadN)
        np.savetxt(v, self.udContribN, fmt="%g")

        udHead = "SCALARS ud_contrib double 1\n" + \
                 "LOOKUP_TABLE default\n"
        v.write(udHead)
        np.savetxt(v, self.udContrib, fmt="%g")

        if (self.defnodeFaultSlipType == '3d'):
            nmHeadE = "SCALARS nm_contrib_east double 1\n" + \
                "LOOKUP_TABLE default\n"
            v.write(nmHeadE)
            np.savetxt(v, self.nmContribE, fmt="%g")

            nmHeadN = "SCALARS nm_contrib_north double 1\n" + \
                "LOOKUP_TABLE default\n"
            v.write(nmHeadN)
            np.savetxt(v, self.nmContribN, fmt="%g")

            nmHead = "SCALARS nm_contrib double 1\n" + \
                "LOOKUP_TABLE default\n"
            v.write(nmHead)
            np.savetxt(v, self.nmContrib, fmt="%g")

        stHead = "SCALARS strike_degrees double 1\n" + \
                 "LOOKUP_TABLE default\n"
        v.write(stHead)
        np.savetxt(v, self.strikeDeg, fmt="%g")

        dpHead = "SCALARS dip_degrees double 1\n" + \
                 "LOOKUP_TABLE default\n"
        v.write(dpHead)
        np.savetxt(v, self.dipDeg, fmt="%g")

        rkeHead = "SCALARS rake_degrees_east double 1\n" + \
                  "LOOKUP_TABLE default\n"
        v.write(rkeHead)
        np.savetxt(v, self.rakeDegE, fmt="%g")

        rknHead = "SCALARS rake_degrees_north double 1\n" + \
                  "LOOKUP_TABLE default\n"
        v.write(rknHead)
        np.savetxt(v, self.rakeDegN, fmt="%g")

        rkHead = "SCALARS rake_degrees double 1\n" + \
            "LOOKUP_TABLE default\n"
        v.write(rkHead)
        np.savetxt(v, self.rakeDeg, fmt="%g")

        v.close()

        return


    def _writeUnmatchedSites(self):
        """
        Function to write coordinates of unmatched sites to a file.
        """

        print("  Writing unmatched site coordinates:")
        sys.stdout.flush()

        outputDir = Path(self.responseInfoRoot).parents[0]
        outputDir.mkdir(parents=True, exist_ok=True)
        if (self.useGps):
            numUnmatchedGPSSites = self.unmatchedGPSSiteCoords.shape[0]
            print("    Number of unmatched GPS sites:  %d" % numUnmatchedGPSSites)
            if (numUnmatchedGPSSites != 0):
                outArr = np.hstack((self.unmatchedGPSSiteCoords, self.unmatchedSiteDistancesGps.reshape(numUnmatchedGPSSites,1)))
                gpsFile = self.responseInfoRoot + "_unmatched_sites_gps.txt"
                gpsHead = 'X\tY\tZ\tClosest_site_dist'
                np.savetxt(gpsFile, outArr, delimiter='\t', header=gpsHead)

        if (self.useInsar):
            numUnmatchedInsarSites = self.unmatchedInsarSiteCoords.shape[0]
            print("    Number of unmatched InSAR sites:  %d" % numUnmatchedInsarSites)
            if (numUnmatchedInsarSites != 0):
                outArr = np.hstack((self.unmatchedInsarSiteCoords, self.unmatchedSiteDistancesInsar.reshape(numUnmatchedInsarSites,1)))
                insarFile = self.responseInfoRoot + "_unmatched_sites_insar.txt"
                insarHead = 'X\tY\tZ\tClosest_site_dist'
                np.savetxt(insarFile, outArr, delimiter='\t', header=insarHead)

        if (self.useUp):
            numUnmatchedUpSites = self.unmatchedUpSiteCoords.shape[0]
            print("    Number of unmatched Up sites:  %d" % numUnmatchedUpSites)
            if (numUnmatchedUpSites != 0):
                outArr = np.hstack((self.unmatchedUpSiteCoords, self.unmatchedSiteDistancesUp.reshape(numUnmatchedUpSites,1)))
                upFile = self.responseInfoRoot + "_unmatched_sites_up.txt"
                upHead = 'X\tY\tZ\tClosest_site_dist'
                np.savetxt(upFile, outArr, delimiter='\t', header=upHead)

        return


    def _writeResponseInfo(self):
        """
        Function to write response information to a VTK file.
        """

        print("  Writing response information:")
        sys.stdout.flush()

        outputDir = Path(self.responseInfoRoot).parents[0]
        outputDir.mkdir(parents=True, exist_ok=True)
        if (self.useGps):
            gpsFile = self.responseInfoRoot + "_gps.vtk"
            vtkHeadG = "# vtk DataFile Version 2.0\n" + \
                       "GPS Response information for PyLith to Defnode conversion.\n" + \
                       "ASCII\n" + \
                       "DATASET POLYDATA\n" + \
                       "POINTS %d double\n" % self.numGpsSites
            zGps = np.zeros((self.numGpsSites, 1), dtype=np.float64)
            gpsOut = np.hstack((self.gpsCoordsCart, zGps))
            g = open(gpsFile, 'w')
            g.write(vtkHeadG)
            np.savetxt(g, gpsOut)
            if (self.unmatchedSiteOption == 'zero'):
                unmatchedHead1 = "POINT_DATA %d\n" % self.numGpsSites
                unmatchedHead = unmatchedHead1 + "SCALARS unmatched_gps_sites int 1\n" + \
                    "LOOKUP_TABLE default\n"
                g.write(unmatchedHead)
                np.savetxt(g, self.unmatchedGPSSites, fmt="%d")
                distHead = "SCALARS gps_dist_to_closest double 1\n" + \
                    "LOOKUP_TABLE default\n"
                g.write(distHead)
                np.savetxt(g, self.siteDistancesGps)
            g.close()

        if (self.useInsar):
            insarFile = self.responseInfoRoot + "_insar.vtk"
            vtkHeadI = "# vtk DataFile Version 2.0\n" + \
                       "InSAR Response information for PyLith to Defnode conversion.\n" + \
                       "ASCII\n" + \
                       "DATASET POLYDATA\n" + \
                       "POINTS %d double\n" % self.numInsarSites
            zInsar = np.zeros((self.numInsarSites, 1), dtype=np.float64)
            insarOut = np.hstack((self.insarCoordsCart, zInsar))
            i = open(insarFile, 'w')
            i.write(vtkHeadI)
            np.savetxt(i, insarOut)
            if (self.unmatchedSiteOption == 'zero'):
                unmatchedHead1 = "POINT_DATA %d\n" % self.numInsarSites
                unmatchedHead = unmatchedHead1 + "SCALARS unmatched_insar_sites int 1\n" + \
                    "LOOKUP_TABLE default\n"
                i.write(unmatchedHead)
                np.savetxt(i, self.unmatchedInsarSites, fmt="%d")
                distHead = "SCALARS insar_dist_to_closest double 1\n" + \
                    "LOOKUP_TABLE default\n"
                i.write(distHead)
                np.savetxt(i, self.siteDistancesInsar)
            i.close()

        if (self.useUp):
            upFile = self.responseInfoRoot + "_up.vtk"
            vtkHeadU = "# vtk DataFile Version 2.0\n" + \
                       "Uplift Response information for PyLith to Defnode conversion.\n" + \
                       "ASCII\n" + \
                       "DATASET POLYDATA\n" + \
                       "POINTS %d double\n" % self.numUpSites
            zUp = np.zeros((self.numUpSites, 1), dtype=np.float64)
            upOut = np.hstack((self.upCoordsCart, zUp))
            u = open(upFile, 'w')
            u.write(vtkHeadU)
            np.savetxt(u, upOut)
            if (self.unmatchedSiteOption == 'zero'):
                unmatchedHead1 = "POINT_DATA %d\n" % self.numUpSites
                unmatchedHead = unmatchedHead1 + "SCALARS unmatched_up_sites int 1\n" + \
                    "LOOKUP_TABLE default\n"
                u.write(unmatchedHead)
                np.savetxt(u, self.unmatchedUpSites, fmt="%d")
                distHead = "SCALARS up_dist_to_closest double 1\n" + \
                    "LOOKUP_TABLE default\n"
                u.write(distHead)
                np.savetxt(u, self.siteDistancesUp)
            u.close()

        return


    def _writeTDefnodeGF(self):
        """
        Function to write out Tdefnode GF.
        """

        print("  Writing TDefnode Green's functions:")
        sys.stdout.flush()

        nodeNum = 0
        faultString = repr(self.defnodeFaultNum).rjust(3, '0')
        gfFortranGPS = '(A1, 2f10.4, 1x, (6d20.13))'
        gfFmtGPS = FortranRecordWriter(gfFortranGPS)
        gPref = 'G'
        iPref = 'I'
        outputDir = Path(self.gfOutputDirectory)
        outputDir.mkdir(parents=True, exist_ok=True)

        # Loop over number of along-strike and downdip nodes.
        for asNode in range(self.numAsNodes):
            asNodeNum = asNode + 1
            asNodeString = repr(asNodeNum).rjust(3, '0')
            for ddNode in range(self.numDdNodes):
                ddNodeNum = ddNode + 1
                ddNodeString = repr(ddNodeNum).rjust(3, '0')
                gOutFile = 'gf' + faultString + asNodeString + ddNodeString + 'g'
                outFileG = os.path.join(self.gfOutputDirectory, gOutFile)
                iOutFile = 'gf' + faultString + asNodeString + ddNodeString + 'i'
                outFileI = os.path.join(self.gfOutputDirectory, iOutFile)
                if (self.useGps):
                    g = open(outFileG, 'w')
                    g.write(self.gpsHeaders[nodeNum] + "\n")
                    for gpsSiteNum in range(self.numGpsSites):
                        outLine = gfFmtGPS.write([gPref, self.gpsCoordsGeog[gpsSiteNum, 0],
                                                  self.gpsCoordsGeog[gpsSiteNum, 1],
                                                  self.defGfG[nodeNum, gpsSiteNum, 0],
                                                  self.defGfG[nodeNum, gpsSiteNum, 1],
                                                  self.defGfG[nodeNum, gpsSiteNum, 2],
                                                  self.defGfG[nodeNum, gpsSiteNum, 3],
                                                  self.defGfG[nodeNum, gpsSiteNum, 4],
                                                  self.defGfG[nodeNum, gpsSiteNum, 5]])
                        g.write(outLine + "\n")

                    g.close()

                if (self.useInsar):
                    i = open(outFileI, 'w')
                    i.write(self.insarHeaders[nodeNum] + "\n")
                    for insarSiteNum in range(self.numInsarSites):
                        outLine = gfFmtGPS.write([iPref, self.insarCoordsGeog[insarSiteNum, 0],
                                                  self.insarCoordsGeog[insarSiteNum, 1],
                                                  self.defGfI[nodeNum, insarSiteNum, 0],
                                                  self.defGfI[nodeNum, insarSiteNum, 1],
                                                  self.defGfI[nodeNum, insarSiteNum, 2],
                                                  self.defGfI[nodeNum, insarSiteNum, 3],
                                                  self.defGfI[nodeNum, insarSiteNum, 4],
                                                  self.defGfI[nodeNum, insarSiteNum, 5]])
                        i.write(outLine + "\n")

                    i.close()

                nodeNum += 1

        return


    def _writeDefnodeGF(self):
        """
        Function to write out defnode GF.
        """

        print("  Writing Defnode Green's functions:")
        sys.stdout.flush()

        nodeNum = 0
        faultString = repr(self.defnodeFaultNum).rjust(3, '0')
        gfFortranGPS = '(A1, 2f10.4, 1x, (4d20.13))'
        gfFortranUp = '(A1, 2f10.4, (2d20.13), 1x, a8)'
        gfFmtGPS = FortranRecordWriter(gfFortranGPS)
        gfFmtUp = FortranRecordWriter(gfFortranUp)
        gPref = 'G'
        uPref = 'U'
        outputDir = Path(self.gfOutputDirectory)
        outputDir.mkdir(parents=True, exist_ok=True)

        # Loop over number of along-strike and downdip nodes.
        for asNode in range(self.numAsNodes):
            asNodeNum = asNode + 1
            asNodeString = repr(asNodeNum).rjust(3, '0')
            for ddNode in range(self.numDdNodes):
                ddNodeNum = ddNode + 1
                ddNodeString = repr(ddNodeNum).rjust(3, '0')
                gOutFile = 'gf' + faultString + asNodeString + ddNodeString + 'g'
                uOutFile = 'gf' + faultString + asNodeString + ddNodeString + 'u'
                outFileG = os.path.join(self.gfOutputDirectory, gOutFile)
                outFileU = os.path.join(self.gfOutputDirectory, uOutFile)
                if (self.useGps):
                    g = open(outFileG, 'w')
                    g.write(self.gpsHeaders[nodeNum] + "\n")
                    for gpsSiteNum in range(self.numGpsSites):
                        outLine = gfFmtGPS.write([gPref, self.gpsCoordsGeog[gpsSiteNum, 0],
                                                  self.gpsCoordsGeog[gpsSiteNum, 1],
                                                  self.defGfG[nodeNum, gpsSiteNum, 0],
                                                  self.defGfG[nodeNum, gpsSiteNum, 1],
                                                  self.defGfG[nodeNum, gpsSiteNum, 2],
                                                  self.defGfG[nodeNum, gpsSiteNum, 3]])
                        g.write(outLine + "\n")

                    g.close()

                if (self.useUp):
                    u = open(outFileU, 'w')
                    u.write(self.upHeaders[nodeNum] + "\n")
                    for upSiteNum in range(self.numUpSites):
                        outLine = gfFmtUp.write([uPref, self.upCoordsGeog[upSiteNum, 0],
                                                 self.upCoordsGeog[upSiteNum, 1],
                                                 self.defGfU[nodeNum, upSiteNum, 0],
                                                 self.defGfU[nodeNum, upSiteNum, 1],
                                                 self.upSites[upSiteNum]])
                        u.write(outLine + "\n")
          
                    u.close()
                nodeNum += 1

        return


    def _bilinearInterp(self, quad, pointCoords):
        """
        Function to perform bilinear interpolation for a quadrilateral.
        """
    
        # Compute coefficients for forward mapping.
        a0 = 0.25 * ((quad[0,0] + quad[1,0]) + (quad[2,0] + quad[3,0]))
        a1 = 0.25 * ((quad[1,0] - quad[0,0]) + (quad[2,0] - quad[3,0]))
        a2 = 0.25 * ((quad[2,0] + quad[3,0]) - (quad[0,0] + quad[1,0]))
        a3 = 0.25 * ((quad[0,0] - quad[1,0]) + (quad[2,0] - quad[3,0]))
        b0 = 0.25 * ((quad[0,1] + quad[1,1]) + (quad[2,1] + quad[3,1]))
        b1 = 0.25 * ((quad[1,1] - quad[0,1]) + (quad[2,1] - quad[3,1]))
        b2 = 0.25 * ((quad[2,1] + quad[3,1]) - (quad[0,1] + quad[1,1]))
        b3 = 0.25 * ((quad[0,1] - quad[1,1]) + (quad[2,1] - quad[3,1]))
        x0 = pointCoords[0] - a0
        y0 = pointCoords[1] - b0

        # Quadratic coefficients.
        A = a3 * b2 - a2 * b3
        B = (x0 * b3 + a1 * b2) - (y0 * a3 + a2 * b1)
        C = x0 * b1 - y0 * a1

        # Try first solution, then other one if value is out of range.
        discr = B * B - 4.0 * A * C
        # t1 = 0.5 * (-B + math.sqrt(discr))/A
        # t2 = 0.5 * (-B - math.sqrt(discr))/A
    
        if (A == 0.0):
            t = -C/B
        else:
            t = 0.5 * (-B + math.sqrt(discr))/A
            if (t < -1.0 - self.bilinearCoeffEpsilon or t > 1.0 + self.bilinearCoeffEpsilon):
                t = 0.5 * (-B - math.sqrt(discr))/A
        if (t < -1.0 - self.bilinearCoeffEpsilon or t > 1.0 + self.bilinearCoeffEpsilon):
            msg1 = "No root found for point %g %g\n" % (pointCoords[0], pointCoords[1])
            msg2 = "Patch coordinates:  %g %g\n" % (quad[0,0], quad[0,1])
            msg3 = "                    %g %g\n" % (quad[1,0], quad[1,1])
            msg4 = "                    %g %g\n" % (quad[2,0], quad[2,1])
            msg5 = "                    %g %g\n" % (quad[3,0], quad[3,1])
            msg = msg1 + msg2 + msg3 + msg4 + msg5
            raise ValueError(msg)

        # Compute other local coordinate.
        s = (x0 - a2 * t)/(a1 + a3 * t)
    
        # Test solution.
        xTest = a0 + a1 * s + a2 * t + a3 * s * t
        yTest = b0 + b1 * s + b2 * t + b3 * s * t
        xDiff = xTest - pointCoords[0]
        yDiff = yTest - pointCoords[1]
        if (math.fabs(xDiff) > self.nodeMatchEpsilon or math.fabs(yDiff) > self.nodeMatchEpsilon):
            msg = "Bilinear interpolation failed for point: %g  %g" % \
                  (pointCoords[0], pointCoords[1])
            raise ValueError(msg)

        # Compute slip.
        sa = np.array([ -1.0,  1.0,  1.0,  -1.0], dtype=np.float64)
        ta = np.array([ -1.0, -1.0,  1.0,   1.0], dtype=np.float64)
        shape = 0.25 * (1.0 + sa * s) * (1.0 + ta * t)
        slip = np.dot(self.quadWeight, shape)

        return slip


    def _coincidentNode(self, point, patchCoords):
        """
        Function to determine whether a PyLith fault vertex is nearly coincident
        with a defnode node.
        ***Currently unused***
        """
        dist = scipy.spatial.distance.cdist(patchCoords, point.reshape((1, 2)))
        coincident = False
        minDiff = np.argmin(dist)
        if (dist[minDiff] <= self.nodeMatchEpsilon):
            coincident = True

        return (coincident, minDiff)


    def _pointOnEdge(self, point, patchCoords):
        """
        Function to determine whether a PyLith fault vertex is nearly on
        a defnode edge.
        ***Currently unused***
        """
        numNodes = 4
        onLine = False
        xMin = np.amin(patchCoords[:,0])
        xMax = np.amax(patchCoords[:,0])
        yMin = np.amin(patchCoords[:,1])
        yMax = np.amax(patchCoords[:,1])
        if (point[0] < xMin - self.nodeMatchEpsilon or point[0] > xMax + self.nodeMatchEpsilon or \
            point[1] < yMin - self.nodeMatchEpsilon or point[1] > yMax + self.nodeMatchEpsilon):
            return onLine
        testPoint = np.zeros(2, dtype=np.float64)
        for nodeNum in range(numNodes):
            n1 = nodeNum
            n2 = n1 + 1
            if (nodeNum == numNodes - 1):
                n2 = 0
            p1 = patchCoords[n1,:]
            p2 = patchCoords[n2,:]
      
        coordsDiff = p2 - p1
        coordsNorm = np.linalg.norm(coordsDiff)
        u = ((point[0] - p1[0]) * (p2[0] - p1[0]) + \
             (point[1] - p1[1]) * (p2[1] - p1[1]))/(coordsNorm * coordsNorm)
        testPoint = p1 + u * (p2 - p1)
        distance = np.linalg.norm(testPoint - point)
        if (distance <= self.nodeMatchEpsilon):
            onLine = True
            return onLine

        return onLine


    def _triArea(self, cellCoords):
        """
        Function to compute the area of a triangle.
        """
        v1 = cellCoords[1,:] - cellCoords[0,:]
        v2 = cellCoords[2,:] - cellCoords[0,:]
        area = 0.5 * np.linalg.norm(np.cross(v1,v2))

        return area


    def _computeLocalPlaneCoords(self, points, rotationMatrix, planeOrigin):
        """
        Compute local plane coordinates, given an origin and transformation matrix.
        """
        # pointsLocal = np.dot(points - planeOrigin, rotationMatrix.transpose())
        pointsLocal = np.dot(points - planeOrigin, rotationMatrix)

        return pointsLocal


    def _fitPlaneToPoints(self, points):
        """
        Find best-fit plane to a set of points.
        """

        # Compute plane origin and subtract it from the points array.
        eps = 1.0e-5
        planeOrigin = np.mean(points, axis=0)
        x = points - planeOrigin

        # Dot product to yield a 3x3 array.
        moment = np.dot(x.T, x)

        # Extract single values from SVD computation to get normal.
        planeNormal = np.linalg.svd(moment)[0][:,-1]
        planeNormal /= np.linalg.norm(planeNormal)
        small = np.where(np.abs(planeNormal) < eps)
        planeNormal[small] = 0.0
        planeNormal /= np.linalg.norm(planeNormal)
        if (planeNormal[-1] < 0.0):
            planeNormal *= -1.0

        return (planeNormal, planeOrigin)


    def _getFaultRotationMatrix(self, planeNormal):
         """
         Compute rotation matrix, given the normal to the plane. If the normal is nearly
         vertical an alternate reference direction is used to compute the two tangential
         directions.
         Returned values are:
             rotation_matrix: 3x3 rotation matrix with columns (tan_dir1, tan_dir2, plane_normal).
         """
         # Reference directions to try are z=1 (vertical) and y=1 (north).
         cutoffVecmag = 0.98
         refDir1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
         refDir2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
         refDir = refDir1

         # If normal is nearly vertical, use north reference direction.
         if (np.dot(refDir1, planeNormal) > cutoffVecmag):
             refDir = refDir2
        
         # Get two tangential directions in plane.
         tanDir1 = np.cross(refDir, planeNormal)
         tanDir1 /= np.linalg.norm(tanDir1)
         tanDir2 = np.cross(planeNormal, tanDir1)
         tanDir2 /= np.linalg.norm(tanDir2)
         
         # Form rotation matrix.
         rotationMatrix = np.column_stack((tanDir1, tanDir2, planeNormal))
         
         return rotationMatrix


    def _getAreas2(self):
        """
        Function to compute areas associated with each PyLith vertex within each
        defnode cell.
        This version projects all nearby triangle centers into the best-fit plane of the
        quadrilateral (loop is over Defnode quadrilaterals).
        """
        print("  Computing areas associated with each vertex:")
        sys.stdout.flush()

        # Define arrays
        self.totalArea = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.numPatchesShared = np.zeros(self.numFaultVerts, dtype=np.int64)
        self.connectedPatches = -1 * np.ones((self.numFaultVerts, 4), dtype=np.int64)
        self.connectedPatchArea = np.zeros((self.numFaultVerts, 4), dtype=np.float64)
        self.patchArea = np.zeros(self.numDefCells, dtype=np.float64)

        printIncr = 100

        # Get coordinates in selected fault plane.
        if (self.faultProjectionPlane == 'best_fit_plane'):
            (planeNormal, planeOrigin) = self._fitPlaneToPoints(self.defNodeCoords)
        elif (self.faultProjectionPlane == 'defnode_endpoints'):
            points = np.zeros((3,3), dtype=np.float64)
            points[0,:] = self.defNodeCoords[0,:]
            points[1,:] = self.defNodeCoords[self.numDefNodes - self.numDdNodes,:]
            points[2,:] = self.defNodeCoords[-1,:]
            (planeNormal, planeOrigin) = self._fitPlaneToPoints(points)
        elif (self.faultProjectionPlane == 'xy_plane'):
            planeNormal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            planeOrigin = np.mean(self.defNodeCoords, axis=0)
        elif (self.faultProjectionPlane == 'xz_plane'):
            planeNormal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            planeOrigin = np.mean(self.defNodeCoords, axis=0)
        elif (self.faultProjectionPlane == 'yz_plane'):
            planeNormal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            planeOrigin = np.mean(self.defNodeCoords, axis=0)
        else:
            msg = "Unknown fault projection plane:  %s" % self.faultProjectionPlane
            raise ValueError(msg)

        rotationMatrix = self._getFaultRotationMatrix(planeNormal)
        self.defNodeCoordsLocal = self._computeLocalPlaneCoords(self.defNodeCoords, rotationMatrix, planeOrigin)
        self.faultCoordsLocal = self._computeLocalPlaneCoords(self.faultCoords, rotationMatrix, planeOrigin)

        # Make KD-tree of local PyLith cell centroids
        numSearch = 10
        defPatchCoordsLocal = self.defNodeCoordsLocal[self.defCellConnect,:]
        defPatchCentersLocal = np.mean(defPatchCoordsLocal, axis=1)
        faultCellCoordsLocal = self.faultCoordsLocal[self.faultConnect,:]
        faultCellCentersLocal = np.mean(faultCellCoordsLocal, axis=1)
        tree = scipy.spatial.cKDTree(defPatchCentersLocal)
        numSearch = min(numSearch, self.numDefCells)

        # Search tree to find closest Defnode patches for each fault vertex.
        (distances, patchesNear) = tree.query(faultCellCentersLocal, k=numSearch)

        # Loop over fault cells.
        for cellNum in range(self.numFaultCells):
            if (cellNum % printIncr == 0):
                print("    Working on cell number %d:" % cellNum)
                sys.stdout.flush()
            cellVerts = self.faultConnect[cellNum,:]
            cellCoords = self.faultCoords[cellVerts,:]
            cellCenterLocal = faultCellCentersLocal[cellNum,:]
            cellArea = self._triArea(cellCoords)
            cellCenterXY = cellCenterLocal[0:2]
            cellPoint = shapely.geometry.Point(cellCenterXY)
            self.totalArea[cellVerts] += cellArea/3.0
            # Loop over closest defnode patches.
            for patch in patchesNear[cellNum, :]:
                patchCoords = self.defNodeCoordsLocal[self.defCellConnect[patch, :],0:2]
                patchPoly = shapely.geometry.Polygon(patchCoords[:,:])
                inPoly = patchPoly.contains(cellPoint)
                if (inPoly):
                    self.patchArea[patch] += cellArea
                    for vertNum in cellVerts:
                        alreadyUsed = np.where(self.connectedPatches[vertNum,:] == patch)
                        if (alreadyUsed[0].shape[0] == 0):
                            patchInd = np.argmin(self.connectedPatches[vertNum,:])
                            self.connectedPatches[vertNum, patchInd] = patch
                            self.connectedPatchArea[vertNum, patchInd] += cellArea/3.0
                            self.numPatchesShared[vertNum] += 1
                        else:
                            self.connectedPatchArea[vertNum, alreadyUsed] += cellArea/3.0
              
                    break
                  
        return


    def _getAreas(self):
        """
        Function to compute areas associated with each PyLith vertex within each
        defnode cell.
        """
        print("  Computing areas associated with each vertex:")
        sys.stdout.flush()

        # Define arrays
        self.totalArea = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.numPatchesShared = np.zeros(self.numFaultVerts, dtype=np.int64)
        self.connectedPatches = -1 * np.ones((self.numFaultVerts, 4), dtype=np.int64)
        self.connectedPatchArea = np.zeros((self.numFaultVerts, 4), dtype=np.float64)
        self.patchArea = np.zeros(self.numDefCells, dtype=np.float64)

        printIncr = 100

        # Get coordinates in selected fault plane.
        if (self.faultProjectionPlane == 'best_fit_plane'):
            (planeNormal, planeOrigin) = self._fitPlaneToPoints(self.defNodeCoords)
        elif (self.faultProjectionPlane == 'defnode_endpoints'):
            points = np.zeros((4,3), dtype=np.float64)
            points[0,:] = self.defNodeCoords[0,:]
            points[1,:] = self.defNodeCoords[self.numDefNodes - self.numDdNodes,:]
            points[2,:] = self.defNodeCoords[-1,:]
            points[3,:] = self.defNodeCoords[self.numDdNodes - 1,:]
            (planeNormal, planeOrigin) = self._fitPlaneToPoints(points)
        elif (self.faultProjectionPlane == 'xy_plane'):
            planeNormal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            planeOrigin = np.mean(self.defNodeCoords, axis=0)
        elif (self.faultProjectionPlane == 'xz_plane'):
            planeNormal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            planeOrigin = np.mean(self.defNodeCoords, axis=0)
        elif (self.faultProjectionPlane == 'yz_plane'):
            planeNormal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            planeOrigin = np.mean(self.defNodeCoords, axis=0)
        else:
            msg = "Unknown fault projection plane:  %s" % self.faultProjectionPlane
            raise ValueError(msg)

        rotationMatrix = self._getFaultRotationMatrix(planeNormal)
        self.defNodeCoordsLocal = self._computeLocalPlaneCoords(self.defNodeCoords, rotationMatrix, planeOrigin)
        self.faultCoordsLocal = self._computeLocalPlaneCoords(self.faultCoords, rotationMatrix, planeOrigin)

        # Make KD-tree of local PyLith cell centroids
        numSearch = 10
        defPatchCoordsLocal = self.defNodeCoordsLocal[self.defCellConnect,:]
        defPatchCentersLocal = np.mean(defPatchCoordsLocal, axis=1)
        faultCellCoordsLocal = self.faultCoordsLocal[self.faultConnect,:]
        faultCellCentersLocal = np.mean(faultCellCoordsLocal, axis=1)
        tree = scipy.spatial.cKDTree(defPatchCentersLocal)
        numSearch = min(numSearch, self.numDefCells)

        # Search tree to find closest Defnode patches for each fault vertex.
        (distances, patchesNear) = tree.query(faultCellCentersLocal, k=numSearch)

        # Loop over fault cells.
        for cellNum in range(self.numFaultCells):
            if (cellNum % printIncr == 0):
                print("    Working on cell number %d:" % cellNum)
                sys.stdout.flush()
            cellVerts = self.faultConnect[cellNum,:]
            cellCoords = self.faultCoords[cellVerts,:]
            cellCenterLocal = faultCellCentersLocal[cellNum,:]
            cellArea = self._triArea(cellCoords)
            cellCenterXY = cellCenterLocal[0:2]
            cellPoint = shapely.geometry.Point(cellCenterXY)
            self.totalArea[cellVerts] += cellArea/3.0
            # Loop over closest defnode patches.
            for patch in patchesNear[cellNum, :]:
                patchCoords = self.defNodeCoordsLocal[self.defCellConnect[patch, :],0:2]
                patchPoly = shapely.geometry.Polygon(patchCoords[:,:])
                inPoly = patchPoly.contains(cellPoint)
                if (inPoly):
                    self.patchArea[patch] += cellArea
                    for vertNum in cellVerts:
                        alreadyUsed = np.where(self.connectedPatches[vertNum,:] == patch)
                        if (alreadyUsed[0].shape[0] == 0):
                            patchInd = np.argmin(self.connectedPatches[vertNum,:])
                            self.connectedPatches[vertNum, patchInd] = patch
                            self.connectedPatchArea[vertNum, patchInd] += cellArea/3.0
                            self.numPatchesShared[vertNum] += 1
                        else:
                            self.connectedPatchArea[vertNum, alreadyUsed] += cellArea/3.0
              
                    break
                  
        return


    def _integrateGF(self):
        """
        Function to perform bilinear interpolation on PyLith GF and sum them to
        obtain defnode GF.
        """

        print("  Integrating PyLith Green's functions:")
        sys.stdout.flush()

        if (self.gfType == 'defnode'):
            if (self.useGps):
                self.defGfG = np.zeros((self.numDefNodes, self.numGpsSites, 4), dtype=np.float64)
            if (self.useUp):
                self.defGfU = np.zeros((self.numDefNodes, self.numUpSites, 2), dtype=np.float64)
        else:
            if (self.useGps):
                self.defGfG = np.zeros((self.numDefNodes, self.numGpsSites, 6), dtype=np.float64)
            if (self.useInsar):
                self.defGfI = np.zeros((self.numDefNodes, self.numInsarSites, 6), dtype=np.float64)
        self.faultSlip = np.zeros(self.numFaultVerts, dtype=np.float64)
        numCorners = 4
        printIncr = 100

        # Loop over PyLith impulses.
        for impulse in range(self.numImpulses):
            if (impulse % printIncr == 0):
                print("    Working on impulse number %d:" % impulse)
                sys.stdout.flush()
            vertNum = self.impulseVerts[impulse]
            faultXY = self.faultCoordsLocal[vertNum,0:2]
            # faultXY = self.faultCoords[vertNum,0:2]
            for patchNum in range(self.numPatchesShared[vertNum]):
                patch = self.connectedPatches[vertNum, patchNum]
                patchCoords = self.defNodeCoordsLocal[self.defCellConnect[patch, :],0:2]
                # patchCoords = self.defNodeCoords[self.defCellConnect[patch, :],0:2]
                areaFac = self.connectedPatchArea[vertNum, patchNum]/self.totalArea[vertNum]
            
                slip = areaFac * self._bilinearInterp(patchCoords, faultXY)
                if (slip.sum() <= 0.0):
                    print("Zero-sum slip for impulse, vertex, patch: %d %d %d" % (impulse, vertNum, patch))
                    sys.stdout.flush()
                for corner in range(numCorners):
                    node = self.defCellConnect[patch, corner]
                    self.faultSlip[vertNum] += slip[corner]
                    if (self.gfType == 'defnode'):
                        if (self.useGps):
                            respEG = slip[corner] * self.pyEastGPSGf[impulse,:,:]
                            respNG = slip[corner] * self.pyNorthGPSGf[impulse,:,:]
                            self.defGfG[node, :, 0:2] += respEG
                            self.defGfG[node, :, 2:4] += respNG
                        if (self.useUp):
                            respEU = slip[corner] * self.pyEastUpGf[impulse,:]
                            respNU = slip[corner] * self.pyNorthUpGf[impulse,:]
                            self.defGfU[node, :, 0] += respEU
                            self.defGfU[node, :, 1] += respNU
                    else:
                        if (self.useGps):
                            respEG = slip[corner] * self.pyEastGPSGf[impulse,:,:]
                            respNG = slip[corner] * self.pyNorthGPSGf[impulse,:,:]
                            self.defGfG[node, :, 0::2] += respEG
                            self.defGfG[node, :, 1::2] += respNG
                        if (self.useInsar):
                            respEI = slip[corner] * self.pyEastInsarGf[impulse,:,:]
                            respNI = slip[corner] * self.pyNorthInsarGf[impulse,:,:]
                            self.defGfI[node, :, 0::2] += respEI
                            self.defGfI[node, :, 1::2] += respNI

        return


    def _getResponseIndices(self):
        """
        Function to find indices for PyLith GF responses corresponding to GPS, InSAR, and uplift defnode results.
        """

        print("    Finding PyLith GF indices corresponding to Defnode data points:")
        sys.stdout.flush()

        responseCoordsCart = self.responseCoords[:,0:2]
        gpsIndices = []
        insarIndices = []
        upIndices = []
        unmatchedGPSSites = []
        unmatchedInsarSites = []
        unmatchedUpSites = []
        unmatchedSiteDistancesGps = []
        unmatchedSiteDistancesInsar = []
        unmatchedSiteDistancesUp = []

        # Find indices for GPS sites.
        if (self.useGps):
            distanceG = scipy.spatial.distance.cdist(responseCoordsCart, self.gpsCoordsCart)
            minIndicesG = np.argmin(distanceG, axis=0)
            self.siteDistancesGps = np.zeros(self.numGpsSites, dtype=np.float64)
            for siteNum in range(self.numGpsSites):
                siteId = minIndicesG[siteNum]
                self.siteDistancesGps[siteNum] = distanceG[siteId, siteNum]
                if (distanceG[siteId, siteNum] < self.siteMatchEpsilon):
                    gpsIndices.append(siteId)
                elif (self.unmatchedSiteOption == 'zero'):
                    gpsIndices.append(-1)
                    unmatchedGPSSites.append(siteNum)
                    unmatchedSiteDistancesGps.append(distanceG[siteId, siteNum])
                else:
                    msg1 = "No matching site found for GPS site # %d\n" % siteNum
                    msg2 = "Geographic coordinates:  %g %g\n" % (self.gpsCoordsGeog[siteNum, 0], self.gpsCoordsGeog[siteNum, 1])
                    msg = msg1 + msg2
                    raise ValueError(msg)
            self.unmatchedGPSSites = -1*np.ones(self.numGpsSites, dtype=np.int64)
            self.unmatchedGPSSites[unmatchedGPSSites] = 1
            self.unmatchedGPSSiteCoords = self.responseCoords[unmatchedGPSSites,:]
            self.unmatchedSiteDistancesGps = np.array(unmatchedSiteDistancesGps)

        # Find indices for InSAR sites.
        if (self.useInsar):
            distanceI = scipy.spatial.distance.cdist(responseCoordsCart, self.insarCoordsCart)
            minIndicesI = np.argmin(distanceI, axis=0)
            self.siteDistancesInsar = np.zeros(self.numInsarSites, dtype=np.float64)
            for siteNum in range(self.numInsarSites):
                siteId = minIndicesI[siteNum]
                self.siteDistancesInsar[siteNum] = distanceI[siteId, siteNum]
                if (distanceI[siteId, siteNum] < self.siteMatchEpsilon):
                    insarIndices.append(siteId)
                elif (self.unmatchedSiteOption == 'zero'):
                    insarIndices.append(-1)
                    unmatchedInsarSites.append(siteNum)
                    unmatchedSiteDistancesInsar.append(distanceI[siteId, siteNum])
                else:
                    msg1 = "No matching site found for InSAR site # %d\n" % siteNum
                    msg2 = "Geographic coordinates:  %g %g\n" % (self.insarCoordsGeog[siteNum, 0], self.insarCoordsGeog[siteNum, 1])
                    msg = msg1 + msg2
                    raise ValueError(msg)
            self.unmatchedInsarSites = -1*np.ones(self.numInsarSites, dtype=np.int64)
            self.unmatchedInsarSites[unmatchedInsarSites] = 1
            self.unmatchedInsarSiteCoords = self.responseCoords[unmatchedInsarSites,:]
            self.unmatchedSiteDistancesInsar = np.array(unmatchedSiteDistancesInsar)
          
        # Find indices for uplift sites.
        if (self.useUp):
            distanceU = scipy.spatial.distance.cdist(responseCoordsCart, self.upCoordsCart)
            minIndicesU = np.argmin(distanceU, axis=0)
            self.siteDistancesUp = np.zeros(self.numUpSites, dtype=np.float64)
            for siteNum in range(self.numUpSites):
                siteId = minIndicesU[siteNum]
                self.siteDistancesUp[siteNum] = distanceU[siteId, siteNum]
                if (distanceU[siteId, siteNum] < self.siteMatchEpsilon):
                    upIndices.append(siteId)
                elif (self.unmatchedSiteOption == 'zero'):
                    upIndices.append(-1)
                    unmatchedUpSites.append(siteNum)
                    unmatchedSiteDistancesUp.append(distanceU[siteId, siteNum])
                else:
                    msg1 = "No matching site found for uplift site # %d\n" % siteNum
                    msg2 = "Geographic coordinates:  %g %g\n" % (self.upCoordsGeog[siteNum, 0], self.upCoordsGeog[siteNum, 1])
                    msg = msg1 + msg2
                    raise ValueError(msg)
            self.unmatchedUpSites = -1*np.ones(self.numUpSites, dtype=np.int64)
            self.unmatchedUpSites[unmatchedUpSites] = 1
            self.unmatchedUpSiteCoords = self.responseCoords[unmatchedUpSites,:]
            self.unmatchedSiteDistancesUp = np.array(unmatchedSiteDistancesUp)
          
        return (gpsIndices, insarIndices, upIndices)
    
    
    def _getResponses(self):
        """
        Function to get response coordinates and values.
        """
        print("    Getting PyLith response coordinates and values:")
        
        responseValsNM = None
        # Open response files and get coordinates.
        dataRLL = h5py.File(self.llResponseFile, 'r')
        self.responseCoords = dataRLL['geometry/vertices'][:]
        self.numResponses = self.responseCoords.shape[0]
        dataRUD = h5py.File(self.udResponseFile, 'r')
        responseCoordsRUD = dataRUD['geometry/vertices'][:]
        if (self.defnodeFaultSlipType == '3d'):
            dataRNM = h5py.File(self.nmResponseFile, 'r')
            responseCoordsRNM = dataRNM['geometry/vertices'][:]

        print("      Correlating left-lateral and updip response coordinates:")
        sys.stdout.flush()
        distanceRCLU = scipy.spatial.distance.cdist(self.responseCoords, responseCoordsRUD)
        minIndicesRCLU = np.argmin(distanceRCLU, axis=1)
        coordsRDiff = self.responseCoords - responseCoordsRUD[minIndicesRCLU,:]
        coordsRNorm = np.linalg.norm(coordsRDiff)
        if (coordsRNorm > self.siteMatchEpsilon):
            msg = "Different coordinates for updip and left-lateral responses!"
            raise ValueError(msg)
        if (self.defnodeFaultSlipType == '3d'):
            print("      Correlating left-lateral and fault-normal response coordinates:")
            sys.stdout.flush()
            distanceRCLN = scipy.spatial.distance.cdist(self.responseCoords, responseCoordsRNM)
            minIndicesRCLN = np.argmin(distanceRCLN, axis=1)
            coordsRDiff = self.responseCoords - responseCoordsRNM[minIndicesRCLN,:]
            coordsRNorm = np.linalg.norm(coordsRDiff)
            if (coordsRNorm > self.siteMatchEpsilon):
                msg = "Different coordinates for fault-normal and left-lateral responses!"
                raise ValueError(msg)

        # Get response values.
        # For some reason, the entire array doesn't seem to load unless I use the read_direct method.
        responseDat = dataRLL['vertex_fields/displacement']
        shape = responseDat.shape
        responseValsLL = np.zeros(shape, dtype=np.float64)
        responseValsUD = np.zeros(shape, dtype=np.float64)
        dataRLL['vertex_fields/displacement'].read_direct(responseValsLL)
        self.numImpulses = responseValsLL.shape[0]
        dataRUD['vertex_fields/displacement'].read_direct(responseValsUD)
        responseValsUD = responseValsUD[:,minIndicesRCLU,:]
        if (self.defnodeFaultSlipType == '3d'):
            responseValsNM = np.zeros(shape, dtype=np.float64)
            dataRNM['vertex_fields/displacement'].read_direct(responseValsNM)
            responseValsNM = responseValsNM[:,minIndicesRCLN,:]

        # Pad arrays with zeroes if we are keeping unmatched sites.
        if (self.unmatchedSiteOption == 'zero'):
            padArray = np.zeros((self.numImpulses, 1, 3), dtype=np.float64)
            responseValsLL = np.append(responseValsLL, padArray, axis=1)
            responseValsUD = np.append(responseValsUD, padArray, axis=1)
            if (self.defnodeFaultSlipType == '3d'):
                responseValsNM = np.append(responseValsNM, padArray, axis=1)

        dataRLL.close()
        dataRUD.close()
        if (self.defnodeFaultSlipType == '3d'):
            dataRNM.close()

        return (responseValsLL, responseValsUD, responseValsNM)


    def _getFaultInfo(self):
        """
        Function to get fault information from PyLith fault info file.
        """
        print("    Getting fault information from PyLith fault info file:")
        
        minIndicesFCLN = None
        # Get fault info from left-lateral fault info file.
        llInfoBase = self.llImpulseFile.rstrip('.h5')
        llImpulseInfoFile = llInfoBase + '_info.h5'
        dataIFLL = h5py.File(llImpulseInfoFile, 'r')
        self.faultCoords = dataIFLL['geometry/vertices'][:]
        self.numFaultVerts = self.faultCoords.shape[0]
        self.faultConnect = np.array(dataIFLL['topology/cells'][:], dtype=np.int64)
        self.numFaultCells = self.faultConnect.shape[0]
        faultNormal = dataIFLL['vertex_fields/normal_dir'][0,:,:]
        faultStrike = dataIFLL['vertex_fields/strike_dir'][0,:,:]
        faultDip = dataIFLL['vertex_fields/dip_dir'][0,:,:]
        dataIFLL.close()
        #************** Experiment to make sure normal points upward. **************
        """
        # Don't use this for now.
        negVert = np.where(faultNormal[:,2] < 0.0)
        if (negVert[0].shape[0] > 0):
            refDir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            faultNormal[negVert,:] *= -1.0
            faultStrike = np.cross(refDir, faultNormal)
            faultStrike /= np.linalg.norm(faultStrike, axis=1).reshape(self.numFaultVerts, 1)
            faultDip = np.cross(faultNormal, faultStrike)
            faultDip /= np.linalg.norm(faultDip, axis=1).reshape(self.numFaultVerts, 1)
        """
        #**************** End experiment. *********************

        # Get fault info from updip fault info file.
        udInfoBase = self.udImpulseFile.rstrip('.h5')
        udImpulseInfoFile = udInfoBase + '_info.h5'
        dataIFUD = h5py.File(udImpulseInfoFile, 'r')
        faultCoordsFUD = dataIFUD['geometry/vertices'][:]

        print("      Correlating left-lateral and updip fault coordinates:")
        sys.stdout.flush()
        distanceFCLU = scipy.spatial.distance.cdist(self.faultCoords, faultCoordsFUD)
        minIndicesFCLU = np.argmin(distanceFCLU, axis=1)
        coordsFDiff = self.faultCoords - faultCoordsFUD[minIndicesFCLU,:]
        coordsFNorm = np.linalg.norm(coordsFDiff)
        if (coordsFNorm > self.siteMatchEpsilon):
            msg = "Different coordinates for updip and left-lateral impulses!"
            raise ValueError(msg)
        dataIFUD.close()

        if (self.defnodeFaultSlipType == '3d'):
            # Get fault info from normal fault info file.
            nmInfoBase = self.nmImpulseFile.rstrip('.h5')
            nmImpulseInfoFile = nmInfoBase + '_info.h5'
            dataIFNM = h5py.File(nmImpulseInfoFile, 'r')
            faultCoordsFNM = dataIFNM['geometry/vertices'][:]

            print("      Correlating left-lateral and fault-normal fault coordinates:")
            sys.stdout.flush()
            distanceFCLN = scipy.spatial.distance.cdist(self.faultCoords, faultCoordsFNM)
            minIndicesFCLN = np.argmin(distanceFCLN, axis=1)
            coordsFDiff = self.faultCoords - faultCoordsFNM[minIndicesFCLN,:]
            coordsFNorm = np.linalg.norm(coordsFDiff)
            if (coordsFNorm > self.siteMatchEpsilon):
                msg = "Different coordinates for fault-normal and left-lateral impulses!"
                raise ValueError(msg)
            dataIFNM.close()

        return (faultNormal, faultStrike, faultDip, minIndicesFCLU, minIndicesFCLN)


    def _initializeFaultArrays(self):
        """
        Function to initialize fault and impulse arrays.
        """
        
        self.llContrib = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.udContrib = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.llContribE = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.udContribE = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.llContribN = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.udContribN = np.zeros(self.numFaultVerts, dtype=np.float64)
        if (self.defnodeFaultSlipType == '3d'):
            self.nmContrib = np.zeros(self.numFaultVerts, dtype=np.float64)
            self.nmContribE = np.zeros(self.numFaultVerts, dtype=np.float64)
            self.nmContribN = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.strikeDeg = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.dipDeg = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.rakeDegE = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.rakeDegN = np.zeros(self.numFaultVerts, dtype=np.float64)
        self.rakeDeg = np.zeros(self.numFaultVerts, dtype=np.float64)
        if (self.gfType == 'defnode'):
            if (self.useGps):
                self.pyEastGPSGf = np.zeros((self.numImpulses, self.numGpsSites, 2), dtype=np.float64)
                self.pyNorthGPSGf = np.zeros((self.numImpulses, self.numGpsSites, 2), dtype=np.float64)
            if (self.useUp):
                self.pyEastUpGf = np.zeros((self.numImpulses, self.numUpSites), dtype=np.float64)
                self.pyNorthUpGf = np.zeros((self.numImpulses, self.numUpSites), dtype=np.float64)
        else:
            if (self.useGps):
                self.pyEastGPSGf = np.zeros((self.numImpulses, self.numGpsSites, 3), dtype=np.float64)
                self.pyNorthGPSGf = np.zeros((self.numImpulses, self.numGpsSites, 3), dtype=np.float64)
            if (self.useInsar):
                self.pyEastInsarGf = np.zeros((self.numImpulses, self.numInsarSites, 3), dtype=np.float64)
                self.pyNorthInsarGf = np.zeros((self.numImpulses, self.numInsarSites, 3), dtype=np.float64)

        return

    def _getImpulses(self):
        """
        Get nonzero impulse values.
        """
        print("    Getting PyLith nonzero impuse values:")

        slipNMNZ = None
        # Read impulses.
        # Trying new method using read_direct.
        dataFLL = h5py.File(self.llImpulseFile, 'r')
        dataFUD = h5py.File(self.udImpulseFile, 'r')
        print("      Reading left-lateral fault impulses:")
        slipDat = dataFLL['vertex_fields/slip']
        shape = slipDat.shape
        slipLL = np.zeros(shape, dtype=np.float64)
        dataFLL['vertex_fields/slip'].read_direct(slipLL)
        slipLL = slipLL[:,:,0]

        print("      Reading updip fault impulses:")
        slipUD = np.zeros(shape, dtype=np.float64)
        dataFUD['vertex_fields/slip'].read_direct(slipUD)
        slipUD = slipUD[:,:,1]
        dataFLL.close()
        dataFUD.close()
        slipLLNZ = slipLL.nonzero()
        slipUDNZ = slipUD.nonzero()

        if (self.defnodeFaultSlipType == '3d'):
            dataFNM = h5py.File(self.nmImpulseFile, 'r')
            print("      Reading fault-normal fault impulses:")
            slipNM = np.zeros(shape, dtype=np.float64)
            dataFNM['vertex_fields/slip'].read_direct(slipNM)
            slipNM = slipNM[:,:,2]
            dataFNM.close()
            slipNMNZ = slipNM.nonzero()

        return (slipLLNZ, slipUDNZ, slipNMNZ)
    
    
    def _readPyGFDefnode(self):
        """
        Function to read PyLith impulses/responses, make sure coordinates match
        for left-lateral and updip components, and create indices for updip
        components.
        This version creates arrays needed for Defnode.
        """

        print("  Reading PyLith Green's functions:")
        sys.stdout.flush()

        # Open response files and get coordinates and values.
        (responseValsLL, responseValsUD, responseValsNM) = self._getResponses()

        # Get indices of desired responses.
        (self.gpsIndices, self.insarIndices, self.upIndices) = self._getResponseIndices()

        # Get fault info from left-lateral fault info file.
        (faultNormal, faultStrike, faultDip, minIndicesFCLU, minIndicesFCLN) = self._getFaultInfo()

        # Create fault and impulse arrays.
        self._initializeFaultArrays()

        printIncr = 100
        Ux = 1.0
        Uy = 1.0
        Uz = 0.0
        backSlip = -1.0

        # Get nonzero impulses.
        (slipLLNZ, slipUDNZ, slipNMNZ) = self._getImpulses()

        print("    Computing Green's functions in Defnode coordinate system:")
        
        for impulse in range(self.numImpulses):
            if (impulse % printIncr == 0):
                print("      Working on impulse number %d:" % impulse)
                sys.stdout.flush()
            impulseNumLL = slipLLNZ[0][impulse]
            if (impulseNumLL != impulse):
                msg = "Impulse # %d does not match LL impulse # %d." % (impulse, impulseNumLL)
                raise ValueError(msg)
            vertNumLL = slipLLNZ[1][impulseNumLL]
            vertNumUD = minIndicesFCLU[vertNumLL]
            impulseNumUD = np.argwhere(slipUDNZ[1][:] == vertNumUD)[0][0]
            self.impulseVerts.append(vertNumLL)

            responseLL = self.gfScale * responseValsLL[impulseNumLL,:,:]
            responseUD = self.gfScale * responseValsUD[impulseNumUD,:,:]
            if (self.defnodeFaultSlipType == '3d'):
                vertNumNM = minIndicesFCLN[vertNumLL]
                impulseNumNM = np.argwhere(slipNMNZ[1][:] == vertNumNM)[0][0]
                responseNM = self.gfScale * responseValsNM[impulseNumNM,:,:]

            if (self.useGps):
                gpsResponseLL = responseLL[self.gpsIndices,0:2]
                gpsResponseUD = responseUD[self.gpsIndices,0:2]
                if (self.defnodeFaultSlipType == '3d'):
                    gpsResponseNM = responseNM[self.gpsIndices,0:2]
            if (self.useUp):
                upResponseLL = responseLL[self.upIndices,2]
                upResponseUD = responseUD[self.upIndices,2]
                if (self.defnodeFaultSlipType == '3d'):
                    upResponseNM = responseNM[self.upIndices,2]
      
            # Get E and N vectors in fault plane.
            # Rob's method (I think).
            normal = faultNormal[self.impulseVerts[impulse],:]
            strike = faultStrike[self.impulseVerts[impulse],:]
            dip = faultDip[self.impulseVerts[impulse],:]
            dipAng = math.acos(dip[2]) - 0.5*math.pi
            strikeAng = math.pi + math.atan2(strike[0], strike[1])
            self.strikeDeg[vertNumLL] = math.degrees(strikeAng)
            self.dipDeg[vertNumLL] = math.degrees(dipAng)
            UxpE = Ux * math.cos(strikeAng)
            UypE = Ux * math.sin(strikeAng)
            UzpE = Uz
            U1E = UypE
            U2E = -UxpE
            U3E = UzpE
            UxpN = -Uy * math.sin(strikeAng)
            UypN = Uy * math.cos(strikeAng)
            UzpN = Uz
            U1N = UypN
            U2N = -UxpN
            U3N = UzpN
            if (self.defnodeFaultSlipType == '3d'):
                U3E = UxpE*math.sin(dipAng)
                U2E *= math.cos(dipAng)
                U3N = UxpN*math.sin(dipAng)
                U2N *= math.cos(dipAng)

            U1 = U1E + U1N
            U2 = U2E + U2N
            U3 = U3E + U3N
            rakeE = math.degrees(math.atan2(U2E, U1E))
            rakeN = math.degrees(math.atan2(U2N, U1N))
            rake = math.degrees(math.atan2(U2, U1))

            self.rakeDegE[vertNumLL] = rakeE
            self.rakeDegN[vertNumLL] = rakeN
            self.rakeDeg[vertNumLL] = rake
            self.llContrib[vertNumLL] = U1
            self.udContrib[vertNumLL] = U2
            self.llContribE[vertNumLL] = U1E
            self.llContribN[vertNumLL] = U1N
            self.udContribE[vertNumLL] = U2E
            self.udContribN[vertNumLL] = U2N
            if (self.defnodeFaultSlipType == '3d'):
                self.nmContrib[vertNumLL] = U3
                self.nmContribE[vertNumLL] = U3E
                self.nmContribN[vertNumLL] = U3N

            if (self.useGps):
                self.pyEastGPSGf[impulse,:,:] = backSlip * (U1E * gpsResponseLL + U2E*gpsResponseUD)
                self.pyNorthGPSGf[impulse,:,:] = backSlip * (U1N * gpsResponseLL + U2N*gpsResponseUD)
                if (self.defnodeFaultSlipType == '3d'):
                    self.pyEastGPSGf[impulse,:,:] = backSlip * (U1E * gpsResponseLL + U2E*gpsResponseUD + U3E*gpsResponseNM)
                    self.pyNorthGPSGf[impulse,:,:] = backSlip * (U1N * gpsResponseLL + U2N*gpsResponseUD + U3N*gpsResponseNM)
            if (self.useUp):
                self.pyEastUpGf[impulse,:] = backSlip * (U1E * upResponseLL + U2E*upResponseUD)
                self.pyNorthUpGf[impulse,:] = backSlip * (U1N * upResponseLL + U2N*upResponseUD)
                if (self.defnodeFaultSlipType == '3d'):
                    self.pyEastUpGf[impulse,:] = backSlip * (U1E * upResponseLL + U2E*upResponseUD + U3E*upResponseNM)
                    self.pyNorthUpGf[impulse,:] = backSlip * (U1N * upResponseLL + U2N*upResponseUD + U3N*upResponseNM)

        numImpulseIndices = len(self.impulseVerts)
        uniqueIndices = set(self.impulseVerts)
        if (numImpulseIndices != self.numImpulses):
            msg = "# of impulse indices (%d) not equal to # of impulses (%d)." % (numImpulseIndices, self.numImpulses)
            raise ValueError(msg)
        if (self.numImpulses != len(uniqueIndices)):
            msg = "# of unique impulse indices (%d) != # of impulses (%d)." % (len(uniqueIndices), self.numImpulses)
            raise ValueError(msg)
              
        return
    
    
    def _readPyGFTDefnode(self):
        """
        Function to read PyLith impulses/responses, make sure coordinates match
        for left-lateral and updip components, and create indices for updip
        components.
        This version creates arrays needed for TDefnode.
        """

        print("  Reading PyLith Green's functions:")
        sys.stdout.flush()

        # Open response files and get coordinates and values.
        (responseValsLL, responseValsUD, responseValsNM) = self._getResponses()

        # Get indices of desired responses.
        (self.gpsIndices, self.insarIndices, self.upIndices) = self._getResponseIndices()

        # Get fault info from left-lateral fault info file.
        (faultNormal, faultStrike, faultDip, minIndicesFCLU, minIndicesFCLN) = self._getFaultInfo()

        # Create fault and impulse arrays.
        self._initializeFaultArrays()
        printIncr = 100
        Ux = 1.0
        Uy = 1.0
        Uz = 0.0
        backSlip = -1.0

        # Get nonzero slip impulses.
        (slipLLNZ, slipUDNZ, slipNMNZ) = self._getImpulses()

        print("    Computing Green's functions in TDefnode coordinate system:")
        
        for impulse in range(self.numImpulses):
            if (impulse % printIncr == 0):
                print("      Working on impulse number %d:" % impulse)
                sys.stdout.flush()
            impulseNumLL = slipLLNZ[0][impulse]
            if (impulseNumLL != impulse):
                msg = "Impulse # %d does not match LL impulse # %d." % (impulse, impulseNumLL)
                raise ValueError(msg)
            vertNumLL = slipLLNZ[1][impulseNumLL]
            vertNumUD = minIndicesFCLU[vertNumLL]
            impulseNumUD = np.argwhere(slipUDNZ[1][:] == vertNumUD)[0][0]
            self.impulseVerts.append(vertNumLL)

            responseLL = self.gfScale * responseValsLL[impulseNumLL,:,:]
            responseUD = self.gfScale * responseValsUD[impulseNumUD,:,:]
            if (self.defnodeFaultSlipType == '3d'):
                vertNumNM = minIndicesFCLN[vertNumLL]
                impulseNumNM = np.argwhere(slipNMNZ[1][:] == vertNumNM)[0][0]
                responseNM = self.gfScale * responseValsNM[impulseNumNM,:,:]

            if (self.useGps):
                gpsResponseLL = responseLL[self.gpsIndices,0:3]
                gpsResponseUD = responseUD[self.gpsIndices,0:3]
                if (self.defnodeFaultSlipType == '3d'):
                    gpsResponseNM = responseNM[self.gpsIndices,0:3]
            if (self.useInsar):
                insarResponseLL = responseLL[self.insarIndices,0:3]
                insarResponseUD = responseUD[self.insarIndices,0:3]
                if (self.defnodeFaultSlipType == '3d'):
                    insarResponseNM = responseNM[self.insarIndices,0:3]
      
            # Get E and N vectors in fault plane.
            # Rob's method (I think).
            normal = faultNormal[self.impulseVerts[impulse],:]
            strike = faultStrike[self.impulseVerts[impulse],:]
            dip = faultDip[self.impulseVerts[impulse],:]
            dipAng = math.acos(dip[2]) - 0.5*math.pi
            strikeAng = math.pi + math.atan2(strike[0], strike[1])
            self.strikeDeg[vertNumLL] = math.degrees(strikeAng)
            self.dipDeg[vertNumLL] = math.degrees(dipAng)
            UxpE = Ux * math.cos(strikeAng)
            UypE = Ux * math.sin(strikeAng)
            UzpE = Uz
            U1E = UypE
            U2E = -UxpE
            U3E = UzpE
            UxpN = -Uy * math.sin(strikeAng)
            UypN = Uy * math.cos(strikeAng)
            UzpN = Uz
            U1N = UypN
            U2N = -UxpN
            U3N = UzpN
            if (self.defnodeFaultSlipType == '3d'):
                U3E = UxpE*math.sin(dipAng)
                U2E *= math.cos(dipAng)
                U3N = UxpN*math.sin(dipAng)
                U2N *= math.cos(dipAng)

            U1 = U1E + U1N
            U2 = U2E + U2N
            U3 = U3E + U3N
            rakeE = math.degrees(math.atan2(U2E, U1E))
            rakeN = math.degrees(math.atan2(U2N, U1N))
            rake = math.degrees(math.atan2(U2, U1))

            self.rakeDegE[vertNumLL] = rakeE
            self.rakeDegN[vertNumLL] = rakeN
            self.rakeDeg[vertNumLL] = rake
            self.llContrib[vertNumLL] = U1
            self.udContrib[vertNumLL] = U2
            self.llContribE[vertNumLL] = U1E
            self.llContribN[vertNumLL] = U1N
            self.udContribE[vertNumLL] = U2E
            self.udContribN[vertNumLL] = U2N
            if (self.defnodeFaultSlipType == '3d'):
                self.nmContrib[vertNumLL] = U3
                self.nmContribE[vertNumLL] = U3E
                self.nmContribN[vertNumLL] = U3N

            if (self.useGps):
                self.pyEastGPSGf[impulse,:,:] = backSlip * (U1E * gpsResponseLL + U2E*gpsResponseUD)
                self.pyNorthGPSGf[impulse,:,:] = backSlip * (U1N * gpsResponseLL + U2N*gpsResponseUD)
                if (self.defnodeFaultSlipType == '3d'):
                    self.pyEastGPSGf[impulse,:,:] = backSlip * (U1E * gpsResponseLL + U2E*gpsResponseUD + U3E*gpsResponseNM)
                    self.pyNorthGPSGf[impulse,:,:] = backSlip * (U1N * gpsResponseLL + U2N*gpsResponseUD + U3N*gpsResponseNM)

            if (self.useInsar):
                self.pyEastInsarGf[impulse,:,:] = backSlip * (U1E * insarResponseLL + U2E*insarResponseUD)
                self.pyNorthInsarGf[impulse,:,:] = backSlip * (U1N * insarResponseLL + U2N*insarResponseUD)
                if (self.defnodeFaultSlipType == '3d'):
                    self.pyEastInsarGf[impulse,:,:] = backSlip * (U1E * insarResponseLL + U2E*insarResponseUD + U3E*insarResponseNM)
                    self.pyNorthInsarGf[impulse,:,:] = backSlip * (U1N * insarResponseLL + U2N*insarResponseUD + U3N*insarResponseNM)

        numImpulseIndices = len(self.impulseVerts)
        uniqueIndices = set(self.impulseVerts)
        if (numImpulseIndices != self.numImpulses):
            msg = "# of impulse indices (%d) not equal to # of impulses (%d)." % (numImpulseIndices, self.numImpulses)
            raise ValueError(msg)
        if (self.numImpulses != len(uniqueIndices)):
            msg = "# of unique impulse indices (%d) != # of impulses (%d)." % (len(uniqueIndices), self.numImpulses)
            raise ValueError(msg)
              
        return

    
    def _readDefGf(self, gfFile, gfType, coordsGeog):
        """
        Function to read DEFNODE Green's function file.
        """
        epsilon = 1.0e-5
        coordsGeogCurrent = None
        if (coordsGeog is not None):
            coordsGeogCurrent = coordsGeog.copy()
        f = open(gfFile, 'r')
        fLines = f.readlines()
        numSites = len(fLines) - 1
        gfFortran = '(A1, 2f10.4, 1x, (4d20.13))'
        if (gfType == 'u'):
            gfFortran = '(A1, 2f10.4, (2d20.13), 1x, a8)'
        gfFmt = FortranRecordReader(gfFortran)
        gfCoordsGeog = np.zeros((numSites, 2), dtype=np.float64)
        sites = []
        pointNum = 0

        for lineNum in range(1, numSites + 1):
            if (gfType == 'g'):
                (type, gfCoordsGeog[pointNum,0], gfCoordsGeog[pointNum,1],
                 gf1, gf2, gf3, gf4) = gfFmt.read(fLines[lineNum])
            else:
                (type, gfCoordsGeog[pointNum,0], gfCoordsGeog[pointNum,1], gf1, gf2, site) = gfFmt.read(fLines[lineNum])
                sites.append(site)
            if (coordsGeogCurrent is not None):
                diff = coordsGeogCurrent - gfCoordsGeog[pointNum,:]
                norm = np.linalg.norm(diff, axis=1)
                indMin = np.argmin(norm)
                if (norm[indMin] > epsilon):
                    coordsGeogCurrent = np.append(coordsGeogCurrent, gfCoordsGeog[pointNum,:].reshape(1,2), axis=0)
            else:
                coordsGeogCurrent = gfCoordsGeog[pointNum,:].reshape(1,2)
            pointNum += 1

        (x, y) = transform(self.projWGS84, self.projPylith, coordsGeogCurrent[:,0], coordsGeogCurrent[:,1])
        coordsCartCurrent = np.column_stack((x, y))
        numTotalSites = coordsGeogCurrent.shape[0]

        if (gfType == 'g'):
            return (numTotalSites, coordsGeogCurrent, coordsCartCurrent)
        else:
            return (numTotalSites, coordsGeogCurrent, coordsCartCurrent, sites)

    
    def _readTDefGf(self, gfFile, coordsGeog):
        """
        Function to read TDEFNODE Green's function file.
        """
        epsilon = 1.0e-5
        coordsGeogCurrent = None
        if (coordsGeog is not None):
            coordsGeogCurrent = coordsGeog.copy()
        f = open(gfFile, 'r')
        fLines = f.readlines()
        numSites = len(fLines) - 1
        gfFortran = '(A1, 2f10.4, 1x, (6d20.13))'
        gfFmt = FortranRecordReader(gfFortran)
        gfCoordsGeog = np.zeros((numSites, 2), dtype=np.float64)
        pointNum = 0

        for lineNum in range(1, numSites + 1):
            (type, gfCoordsGeog[pointNum,0], gfCoordsGeog[pointNum,1], gf1, gf2, gf3, gf4, gf5, gf6) = gfFmt.read(fLines[lineNum])
            if (coordsGeogCurrent is not None):
                diff = coordsGeogCurrent - gfCoordsGeog[pointNum,:]
                norm = np.linalg.norm(diff, axis=1)
                indMin = np.argmin(norm)
                if (norm[indMin] > epsilon):
                    coordsGeogCurrent = np.append(coordsGeogCurrent, gfCoordsGeog[pointNum,:].reshape(1,2), axis=0)
            else:
                coordsGeogCurrent = gfCoordsGeog[pointNum,:].reshape(1,2)
            pointNum += 1

        (x, y) = transform(self.projWGS84, self.projPylith, coordsGeogCurrent[:,0], coordsGeogCurrent[:,1])
        coordsCartCurrent = np.column_stack((x, y))
        numTotalSites = coordsGeogCurrent.shape[0]

        return (numTotalSites, coordsGeogCurrent, coordsCartCurrent)


    def _readDefnode(self):
        """
        Read Defnode information from GF files.
        """

        print("  Reading Defnode GF files:")
        sys.stdout.flush()

        # Read first line of each GF file and determine number of along-strike
        # and downdip nodes.
        totalGfPath = os.path.normpath(os.path.join(os.getcwd(), self.defnodeGfDir))
        pref = 'gf' + repr(self.defnodeFaultNum).rjust(3, '0')
        searchGps = os.path.join(totalGfPath, pref + '*g')
        searchInsar = os.path.join(totalGfPath, pref + '*i')
        searchUp = os.path.join(totalGfPath, pref + '*u')
        gpsList = glob.glob(searchGps)
        gpsList.sort()
        upList = glob.glob(searchUp)
        upList.sort()
        insarList = glob.glob(searchInsar)
        insarList.sort()
        for fileNum in range(len(gpsList)):
            f = open(gpsList[fileNum], 'r')
            self.gpsHeaders.append(f.readline())
            f.close()
        for fileNum in range(len(insarList)):
            f = open(insarList[fileNum], 'r')
            self.insarHeaders.append(f.readline())
            f.close()
        for fileNum in range(len(upList)):
            f = open(upList[fileNum], 'r')
            self.upHeaders.append(f.readline())
            f.close()
        self.numGpsHeaders = len(self.gpsHeaders)
        self.numInsarHeaders = len(self.insarHeaders)
        self.numUpHeaders = len(self.upHeaders)
        self.numDefNodes = max(self.numGpsHeaders, self.numInsarHeaders, self.numUpHeaders)
        if (self.numGpsHeaders == 0):
            self.useGps = False
            self.numGpsSites = 0
        if (self.numInsarHeaders == 0):
            self.useInsar = False
            self.numInsarSites = 0
        if (self.numUpHeaders == 0):
            self.useUp = False
            self.numUpSites = 0

        # Coordinate arrays for Defnode.
        defNodeCoordsGeog = np.zeros((self.numDefNodes, 3), dtype=np.float64)
        self.defNodeCoords = np.zeros((self.numDefNodes, 3), dtype=np.float64)

        # Loop over Defnode nodes.
        for lineNum in range(self.numDefNodes):
            if (self.useGps):
                line = self.gpsHeaders[lineNum]
                (type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                 kish, nlayers, gfVersion, dateMade, moment, gpsNear) = self.gfHeadFmtR.read(line)
            if (self.useInsar):
                line = self.insarHeaders[lineNum]
                (type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                 kish, nlayers, gfVersion, dateMade, moment, gpsNear) = self.gfHeadFmtR.read(line)
            if (self.useUp):
                line = self.upHeaders[lineNum]
                (type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                 kish, nlayers, gfVersion, dateMade, moment, gpsNear) = self.gfHeadFmtR.read(line)
            elev *= -1000.0
            defNodeCoordsGeog[lineNum, 0] = lon
            defNodeCoordsGeog[lineNum, 1] = lat
            defNodeCoordsGeog[lineNum, 2] = elev
            self.numAsNodes = max(asNode, self.numAsNodes)
            self.numDdNodes = max(ddNode, self.numDdNodes)

        # Convert to Cartesian coordinates.
        (x, y, z) = transform(self.projWGS84, self.projPylith, defNodeCoordsGeog[:,0], defNodeCoordsGeog[:,1], defNodeCoordsGeog[:,2])
        self.defNodeCoords[:,0] = x
        self.defNodeCoords[:,1] = y
        self.defNodeCoords[:,2] = z

        self._createDefnodeConnect()

        # Get unique observation locations.
        for nodeNum in range(self.numDefNodes):
            if (self.gfType == 'defnode'):
                if (self.useGps):
                    (self.numGpsSites, self.gpsCoordsGeog, self.gpsCoordsCart) = self._readDefGf(gpsList[nodeNum], 'g', self.gpsCoordsGeog)
                if (self.useUp):
                    (self.numUpSites, self.upCoordsGeog, self.upCoordsCart, self.upSites) = self._readDefGf(upList[nodeNum], 'u', self.upCoordsGeog)
            else:
                if (self.useGps):
                    (self.numGpsSites, self.gpsCoordsGeog, self.gpsCoordsCart) = self._readTDefGf(gpsList[nodeNum], self.gpsCoordsGeog)
                if (self.useInsar):
                    (self.numInsarSites, self.insarCoordsGeog, self.insarCoordsCart) = self._readTDefGf(insarList[nodeNum], self.upCoordsGeog)

        # Sort sites by lon, then lat.
        if self.useGps:
            inds = np.lexsort((self.gpsCoordsGeog[:,1], self.gpsCoordsGeog[:,0]))
            self.gpsCoordsGeog = self.gpsCoordsGeog[inds,:]
            self.gpsCoordsCart = self.gpsCoordsCart[inds,:]
        if self.useUp:
            inds = np.lexsort((self.upCoordsGeog[:,1], self.upCoordsGeog[:,0]))
            self.upCoordsGeog = self.upCoordsGeog[inds,:]
            self.upCoordsCart = self.upCoordsCart[inds,:]
        if self.useInsar:
            inds = np.lexsort((self.insarCoordsGeog[:,1], self.insarCoordsGeog[:,0]))
            self.insarCoordsGeog = self.insarCoordsGeog[inds,:]
            self.insarCoordsCart = self.insarCoordsCart[inds,:]
        self._fixDefnodeHeaders()

        return


    def _fixDefnodeHeaders(self):
        """
        Set number of sites to be equal to the maximum for all Defnode/TDefnode GF.
        """

        for defNode in range(self.numDefNodes):
            if (self.useGps):
                line = self.gpsHeaders[defNode]
                (type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                 kish, nlayers, gfVersion, dateMade, moment, gpsNear) = self.gfHeadFmtR.read(line)
                numGF = self.numGpsSites
                line2 = self.gfHeadFmtW.write([type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                                               kish, nlayers, gfVersion, dateMade, moment, gpsNear])
                self.gpsHeaders[defNode] = line2
            if (self.useInsar):
                line = self.insarHeaders[defNode]
                (type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                 kish, nlayers, gfVersion, dateMade, moment, gpsNear) = self.gfHeadFmtR.read(line)
                numGF = self.numInsarSites
                line2 = self.gfHeadFmtW.write([type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                                               kish, nlayers, gfVersion, dateMade, moment, gpsNear])
                self.insarHeaders[defNode] = line2
            if (self.useUp):
                line = self.upHeaders[defNode]
                (type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                 kish, nlayers, gfVersion, dateMade, moment, gpsNear) = self.gfHeadFmtR.read(line)
                numGF = self.numUpSites
                line2 = self.gfHeadFmtW.write([type, kf, asNode, ddNode, numGF, lon, lat, elev, xInterp, wInterp,
                                               kish, nlayers, gfVersion, dateMade, moment, gpsNear])
                self.upHeaders[defNode] = line2

        return


    def _createDefnodeConnect(self):
        """
        Create connectivities based on number of along-strike and downdip nodes.
        """
        numDdCells = self.numDdNodes - 1
        numAsCells = self.numAsNodes - 1
        self.numDefCells = numDdCells*numAsCells
        self.defCellConnect = np.zeros((self.numDefCells, 4), dtype=np.int64)
        cellNum = 0

        for asCell in range(numAsCells):
            for ddCell in range(numDdCells):
                self.defCellConnect[cellNum, 0] = ddCell + asCell*self.numDdNodes
                self.defCellConnect[cellNum, 1] = self.defCellConnect[cellNum, 0] + 1
                self.defCellConnect[cellNum, 2] = self.defCellConnect[cellNum, 1] + self.numDdNodes
                self.defCellConnect[cellNum, 3] = self.defCellConnect[cellNum, 2] - 1
                cellNum += 1
        
        return
    
    
# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = Py2Def()
    app.run()

# End of file
