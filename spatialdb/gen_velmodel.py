#!/usr/bin/env python

"""
Python script to generate spatial database for PyLith based on
velocity model from Rui. Outside of this region, results are
merged into PREM.
"""

import numpy
import scipy.interpolate
from pyproj import Proj
from pyproj import transform
from spatialdata.geocoords.CSGeoProj import CSGeoProj
from spatialdata.geocoords.Converter import convert
from spatialdata.spatialdb.SimpleGridAscii import SimpleGridAscii
from coordsys import cs_geo
from coordsys import cs_meshTibet

import pdb
pdb.set_trace()

# Filenames.
velModelFile = 'liu_etal_2021_SRL_0.5x0.5_wrt_surface_VpVs.txt'
premFile = 'prem_1s.csv'
outSpatialDb = 'merged_velmodel.spatialdb'
outVTK = 'merged_velmodel.vtk'

# Mesh boundaries.
meshLonMin = 90.0
meshLonMax = 114.0
meshLatMin = 16.0
meshLatMax = 39.0
meshElevMin = -402000.0
meshElevMax = 2000.0

# Velocity model dimensions.
numVX = 23
numVY = 27
numVZ = 12
numV = numVX*numVY*numVZ

# Amount to extend each direction.
numAddHoriz = 8
numAddZMin = 6
numAddZMax = 1

# Interpolation distance in each direction.
numInterpHoriz = 3
numInterpVert = 3

# Projections.
WGS84 = "+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0"
TM = "+proj=tmerc +lon_0=102.5 +lat_0=28.0 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0"
projWGS84 = Proj(WGS84)
projTM = Proj(TM)

# Coordinate systems.
csIn = cs_geo()
csMesh = cs_meshTibet()

# Function for Brocher (2005) density/Vp relation.
#-------------------------------------------------------------------------------
def brocherDensity(vp):
    """
    Function to compute density from Vp.
    """
    c1 = 1.6612
    c2 = -0.4721
    c3 = 0.0671
    c4 = -0.0043
    c5 = 0.000106
    density = c1*vp + c2*vp*vp + c3*vp*vp*vp + c4*vp*vp*vp*vp + c5*vp*vp*vp*vp*vp

    return density


def createLogRange(v1, v2, numAdd, refVal):
    """
    Function to generate a geometric progression of values.
    The point order is:
    v1:  Closest to mesh center.
    v2:  Furthest from mesh center.
    """
    dv = v2 - v1
    vrange = numpy.geomspace(refVal, numpy.abs(dv) + refVal, num=numAdd)
    if (dv < 0.0):
        vLRange = numpy.flipud(v1 + refVal - vrange)
    else:
        vLRange = v1 - refVal + vrange

    return vLRange


def writeVtk(file, coords):
    """
    Function to write VTK file of coordinates.
    """
    numPoints = coords.shape[0]
  
    v = open(file, 'w')
    v.write(vtkHead)
    v.write("POINTS %d double\n" % numPoints)
    numpy.savetxt(v, coords, fmt="%15.11e", delimiter='\t')
    connect = numpy.arange(numPoints, dtype=numpy.int)
    numCellVerts = numpy.ones(numPoints, dtype=numpy.int)
    outConnect = numpy.column_stack((numCellVerts, connect))
    v.write("VERTICES %d %d\n" % (numPoints, 2 * numPoints))
    numpy.savetxt(v, outConnect, fmt="%d")
    v.close()

    return


#-------------------------------------------------------------------------------
# Read velocity model files.
vDat = numpy.loadtxt(velModelFile, dtype=numpy.float64)
vLons = vDat[0:numVX,0]
vLats = vDat[0:numVX*numVY:numVX,1]
vElevs = -1000.0*vDat[0::numVX*numVY,2]
vCoordsGeog = vDat[:,0:3]
vCoordsGeog[:,2] *= -1000.0
vVp = vDat[:,3]
vVs = vDat[:,4]
vDensity = brocherDensity(vVp)
westLonsAdd = createLogRange(vLons[0], meshLonMin, numAddHoriz + 1, 10.0)
eastLonsAdd = createLogRange(vLons[-1], meshLonMax, numAddHoriz + 1, 10.0)
southLatsAdd = createLogRange(vLats[0], meshLatMin, numAddHoriz + 1, 10.0)
northLatsAdd = createLogRange(vLats[-1], meshLatMax, numAddHoriz + 1, 10.0)
bottomElevsAdd = createLogRange(vElevs[-1], meshElevMin, numAddZMin + 1, 10000.0)
topElevsAdd = numpy.array([meshElevMax], dtype=numpy.float64)

(xCart, yCart, zCart) = transform(projWGS84, projTM, vCoordsGeog[:,0], vCoordsGeog[:,1], vCoordsGeog[:,2])
vCoordsCart = numpy.column_stack((xCart, yCart, zCart))
vCoordsCartGrid = vCoordsCart.reshape((numVX, numVY, numVZ, 3), order='F')

vCoordsGeogGrid = vCoordsGeog.reshape((numVX, numVY, numVZ, 3), order='F')
vVpGrid = vVp.reshape((numVX, numVY, numVZ), order='F')
vVsGrid = vVs.reshape((numVX, numVY, numVZ), order='F')
vDensityGrid = vDensity.reshape((numVX, numVY, numVZ), order='F')

# New ranges.
lonsNew = numpy.concatenate((westLonsAdd, vLons[1:-1], eastLonsAdd))
latsNew = numpy.concatenate((southLatsAdd, vLats[1:-1], northLatsAdd))
elevsNew = numpy.flipud(numpy.concatenate((bottomElevsAdd, numpy.flipud(vElevs[0:-2]), topElevsAdd)))

# Extended dimensions.
numVXExt = lonsNew.shape[0]
numVYExt = latsNew.shape[0]
numVZExt = elevsNew.shape[0]

# New grid.
(zz, yy, xx) = numpy.meshgrid(elevsNew, latsNew, lonsNew, indexing='ij')
coordsNewGeog = numpy.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
(xNewCart, yNewCart, zNewCart) = transform(projWGS84, projTM, xx.flatten(), yy.flatten(), zz.flatten())
coordsNewCart = numpy.column_stack((xNewCart, yNewCart, zNewCart))
coordsNewGeogGrid = coordsNewGeog.reshape((numVXExt, numVYExt, numVZExt, 3), order='F')
coordsNewCartGrid = coordsNewCart.reshape((numVXExt, numVYExt, numVZExt, 3), order='F')

# PREM.
pDat = numpy.loadtxt(premFile, delimiter=',', dtype=numpy.float64)
numP = pDat.shape[0] + 1
pElevs = numpy.zeros(numP, dtype=numpy.float64)
pVp = numpy.zeros(numP, dtype=numpy.float64)
pVs = numpy.zeros(numP, dtype=numpy.float64)
pDensity = numpy.zeros(numP, dtype=numpy.float64)
pElevs[1:] = -1000.0*pDat[:,1]
pDensity[1:] = pDat[:,2]
pVp[1:] = 0.5*(pDat[:,3] + pDat[:,4])
pVs[1:] = 0.5*(pDat[:,5] + pDat[:,6])

# Correct near-surface for PREM.
pElevs[0] = meshElevMax
pDensity[0:5] = pDensity[5]
pVp[0:5] = pVp[5]
pVs[0:5] = pVs[5]

# Interpolate to elevations of velocity model.
pVpf = scipy.interpolate.interp1d(pElevs, pVp)
pVsf = scipy.interpolate.interp1d(pElevs, pVs)
pDensityf = scipy.interpolate.interp1d(pElevs, pDensity)
pVpInterp = pVpf(elevsNew)
pVsInterp = pVsf(elevsNew)
pDensityInterp = pDensityf(elevsNew)

# Define indices for original velocity model, PREM, and interpolated values.
xIndsOrig = numpy.arange(numAddHoriz, numAddHoriz + numVX, step=1)
yIndsOrig = numpy.arange(numAddHoriz, numAddHoriz + numVY, step=1)
zIndsOrig = numpy.arange(0, numVZ, step=1)
#****************** Finish fixing from here. Might be easier to use logical functions (based on location) rather than indexing.
