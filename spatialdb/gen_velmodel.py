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

# Filenames.
velModelFile = 'liu_etal_2021_SRL_0.5x0.5_wrt_surface_VpVs.txt'
premFile = 'prem_1s.csv'
outSpatialDb = 'merged_velmodel.spatialdb'
outOrigVTK = 'original_velmodel.vtk'
outMergedVTK = 'merged_velmodel.vtk'
outPremVTK = 'prem_velmodel.vtk'

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
# Note that this includes an extra surface layer above z=0.
numVZ = 13
numV = numVX*numVY*numVZ

# Amount to extend each direction.
numAddHoriz = 8
numAddZMin = 6

# Interpolation distance in each direction.
numInterpHoriz = 3
numInterpVert = 3

# Projections.
WGS84 = "+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0"
TM = "+proj=tmerc +lon_0=102.5 +lat_0=28.0 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0"
projWGS84 = Proj(WGS84)
projTM = Proj(TM)

# Coordinate systems.
csGeo = cs_geo()

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


def writeVtk(file, coords, nx, ny, nz, vp, vs, density, meshPart):
    """
    Function to write VTK file.
    """
    # Dimensions.
    numPoints = coords.shape[0]
    numVertsPerCell = 8
    numCellX = nx - 1
    numCellY = ny - 1
    numCellZ = nz - 1
    numCells = numCellX*numCellY*numCellZ

    # Connectivity array.
    connect = []
    numPlane = nx*ny
    for cellZ in range(numCellZ):
        for cellY in range(numCellY):
            for cellX in range(numCellX):
                l1 = cellX + cellY*nx + cellZ*numPlane
                l2 = l1 + 1
                l3 = l2 + nx
                l4 = l1 + nx
                l5 = l1 + numPlane
                l6 = l2 + numPlane
                l7 = l3 + numPlane
                l8 = l4 + numPlane
                cell = [numVertsPerCell, l1, l2, l3, l4, l5, l6, l7, l8]
                connect.append(cell)

    connectArr = numpy.array(connect, dtype=numpy.int64)
    cellTypes = 12*numpy.ones(numCells, dtype=numpy.int64)

    # VTK header.
    vtkHead = "# vtk DataFile Version 2.0\n" + \
        "Velocity model\n" + \
        "ASCII\n" + \
        "DATASET UNSTRUCTURED_GRID\n" + \
        "POINTS %d double\n" % numPoints
  
    # Write coordinates.
    v = open(file, 'w')
    v.write(vtkHead)
    numpy.savetxt(v, coords, fmt="%15.11e", delimiter='\t')

    # Write connectivity.
    numCellEntries = numCells*9
    v.write("CELLS %d %d\n" % (numCells, numCellEntries))
    numpy.savetxt(v, connectArr, fmt="%d")
    v.write("CELL_TYPES %d\n" % numCells)
    numpy.savetxt(v, cellTypes, fmt="%d")

    # Write data fields.
    v.write("POINT_DATA %d\n" % numPoints)
    v.write("SCALARS vp double 1\n")
    v.write("LOOKUP_TABLE default\n")
    numpy.savetxt(v, vp)
    v.write("SCALARS vs double 1\n")
    v.write("LOOKUP_TABLE default\n")
    numpy.savetxt(v, vs)
    v.write("SCALARS density double 1\n")
    v.write("LOOKUP_TABLE default\n")
    numpy.savetxt(v, density)
    v.write("SCALARS mesh_part double 1\n")
    v.write("LOOKUP_TABLE default\n")
    numpy.savetxt(v, meshPart)

    v.close()

    return


#-------------------------------------------------------------------------------
# Read velocity model files.
print("Reading original velocity model:")
vDat = numpy.loadtxt(velModelFile, dtype=numpy.float64)
vDat[:,2] *= -1000.0

# Create fake layer above z=0.
vDatExt = numpy.vstack((vDat[:numVX*numVY,:], vDat))
vDatExt[:numVX*numVY,2] = meshElevMax

# Reorder array so z goes from min to max.
vDatFlipped = numpy.zeros_like(vDatExt)
sliceSize = numVX*numVY
for sliceNum in range(numVZ):
    indOrig = (numVZ - sliceNum - 1)*sliceSize
    indFlipped = sliceNum*sliceSize
    vDatFlipped[indFlipped:indFlipped+sliceSize] = vDatExt[indOrig:indOrig+sliceSize]

vLons = vDatFlipped[0:numVX,0]
vLats = vDatFlipped[0:numVX*numVY:numVX,1]
vElevs = vDatFlipped[0::numVX*numVY,2]
vCoordsGeog = vDatFlipped[:,0:3]
vVp = vDatFlipped[:,3]
vVs = vDatFlipped[:,4]
vDensity = brocherDensity(vVp)
westLonsAdd = createLogRange(vLons[0], meshLonMin, numAddHoriz + 1, 10.0)
eastLonsAdd = createLogRange(vLons[-1], meshLonMax, numAddHoriz + 1, 10.0)
southLatsAdd = createLogRange(vLats[0], meshLatMin, numAddHoriz + 1, 10.0)
northLatsAdd = createLogRange(vLats[-1], meshLatMax, numAddHoriz + 1, 10.0)
bottomElevsAdd = createLogRange(vElevs[0], meshElevMin, numAddZMin + 1, 10000.0)

# New ranges.
lonsNew = numpy.concatenate((westLonsAdd, vLons[1:-1], eastLonsAdd))
latsNew = numpy.concatenate((southLatsAdd, vLats[1:-1], northLatsAdd))
elevsNew = numpy.concatenate((bottomElevsAdd, vElevs[1:]))

# Extended dimensions.
numVXExt = lonsNew.shape[0]
numVYExt = latsNew.shape[0]
numVZExt = elevsNew.shape[0]
numExt = numVXExt*numVYExt*numVZExt

# New grid.
(zz, yy, xx) = numpy.meshgrid(elevsNew, latsNew, lonsNew, indexing='ij')
coordsNewGeog = numpy.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
(xNewCart, yNewCart, zNewCart) = transform(projWGS84, projTM, xx.flatten(), yy.flatten(), zz.flatten())
coordsNewCart = numpy.column_stack((xNewCart, yNewCart, zNewCart))

# PREM.
print("Reading PREM velocity model:")
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

# Create interpolation functions for PREM.
pVpf = scipy.interpolate.interp1d(pElevs, pVp)
pVsf = scipy.interpolate.interp1d(pElevs, pVs)
pDensityf = scipy.interpolate.interp1d(pElevs, pDensity)

print("Computing logical arrays for interpolation:")
# Define indices for original velocity model, PREM, and interpolated values.
xOrig = numpy.logical_and(coordsNewGeog[:,0] >= vLons[0], coordsNewGeog[:,0] <= vLons[-1])
yOrig = numpy.logical_and(coordsNewGeog[:,1] >= vLats[0], coordsNewGeog[:,1] <= vLats[-1])
zOrig = coordsNewGeog[:,2] >= vElevs[0]
indsOrig = xOrig*yOrig*zOrig
coordsOrig = coordsNewCart[indsOrig,:]

# Define indices for interpolated region.
numOuterHoriz = numAddHoriz - numInterpHoriz
numOuterVert = numAddZMin - numInterpVert
xInner = numpy.logical_and(coordsNewGeog[:,0] >= lonsNew[numOuterHoriz], coordsNewGeog[:,0] <= lonsNew[-numOuterHoriz-1])
yInner = numpy.logical_and(coordsNewGeog[:,1] >= latsNew[numOuterHoriz], coordsNewGeog[:,1] <= latsNew[-numOuterHoriz-1])
zInner = coordsNewGeog[:,2] >= elevsNew[numOuterVert]
indsInner = xInner*yInner*zInner
indsInterp = numpy.logical_xor(indsOrig, indsInner)
coordsInterp = coordsNewCart[indsInterp,:]

# Define indices of outer parts of velocity model.
indsOuter = numpy.logical_xor(indsInner, numpy.ones_like(indsInner))
coordsOuter = coordsNewCart[indsOuter,:]
vpOuter = pVpf(coordsOuter[:,2])
vsOuter = pVsf(coordsOuter[:,2])
densityOuter = pDensityf(coordsOuter[:,2])

# Create RBF interpolation functions.
refCoords = numpy.vstack((coordsOrig, coordsOuter))
refVp = numpy.concatenate((vVp, vpOuter))
refVs = numpy.concatenate((vVs, vsOuter))
refDensity = numpy.concatenate((vDensity, densityOuter))

print("Computing interpolation functions:")
# rbfVp = scipy.interpolate.Rbf(refCoords[:,0], refCoords[:,1], refCoords[:,2], refVp, function='linear', smooth=0.1)
# rbfVs = scipy.interpolate.Rbf(refCoords[:,0], refCoords[:,1], refCoords[:,2], refVs, function='linear', smooth=0.1)
# rbfDensity = scipy.interpolate.Rbf(refCoords[:,0], refCoords[:,1], refCoords[:,2], refDensity, function='linear', smooth=0.1)
interpVp = scipy.interpolate.LinearNDInterpolator(refCoords, refVp)
interpVs = scipy.interpolate.LinearNDInterpolator(refCoords, refVs)
interpDensity = scipy.interpolate.LinearNDInterpolator(refCoords, refDensity)

# Compute interpolated values.
print("Applying interpolation functions:")
# vpInterp = rbfVp(coordsInterp[:,0], coordsInterp[:,1], coordsInterp[:,2])
# vsInterp = rbfVs(coordsInterp[:,0], coordsInterp[:,1], coordsInterp[:,2])
# densityInterp = rbfDensity(coordsInterp[:,0], coordsInterp[:,1], coordsInterp[:,2])
vpInterp = interpVp(coordsInterp)
vsInterp = interpVs(coordsInterp)
densityInterp = interpDensity(coordsInterp)

# Merged velocity model.
vpMerged = numpy.zeros_like(coordsNewGeog[:,0])
vsMerged = numpy.zeros_like(coordsNewGeog[:,0])
densityMerged = numpy.zeros_like(coordsNewGeog[:,0])
vpMerged[indsOrig] = vVp
vsMerged[indsOrig] = vVs
densityMerged[indsOrig] = vDensity
vpMerged[indsOuter] = vpOuter
vsMerged[indsOuter] = vsOuter
densityMerged[indsOuter] = densityOuter
vpMerged[indsInterp] = vpInterp
vsMerged[indsInterp] = vsInterp
densityMerged[indsInterp] = densityInterp

# Write velocity model to spatialdb.
print("Writing results:")
writer = SimpleGridAscii()
writer.inventory.filename = outSpatialDb
writer._configure()
values = [{'name': "vp",
           'units': "km/s",
           'data': vpMerged},
          {'name': "vs",
           'units': "km/s",
           'data': vsMerged},
          {'name': "density",
           'units': "kg/m**3",
           'data': densityMerged}]
writer.write({'points': coordsNewGeog,
              'x': lonsNew,
              'y': latsNew,
              'z': elevsNew,
              'coordsys': csGeo,
              'data_dim': 3,
              'values': values})

# Write VTK files of different velocity models.
(xCart, yCart, zCart) = transform(projWGS84, projTM, vCoordsGeog[:,0], vCoordsGeog[:,1], vCoordsGeog[:,2])
vCoordsCart = numpy.column_stack((xCart, yCart, zCart))
meshPart = numpy.ones_like(vVp)
writeVtk(outOrigVTK, vCoordsCart, numVX, numVY, numVZ, vVp, vVs, vDensity, meshPart)

meshPart = numpy.zeros_like(vpMerged)
meshPart[indsOrig] = 1.0
meshPart[indsOuter] = 2.0
meshPart[indsInterp] = 3.0
writeVtk(outMergedVTK, coordsNewCart, numVXExt, numVYExt, numVZExt, vpMerged, vsMerged, densityMerged, meshPart)

vpPrem = pVpf(coordsNewCart[:,2])
vsPrem = pVsf(coordsNewCart[:,2])
densityPrem = pDensityf(coordsNewCart[:,2])
writeVtk(outPremVTK, coordsNewCart, numVXExt, numVYExt, numVZExt, vpPrem, vsPrem, densityPrem, meshPart)
