#!/usr/bin/env python

"""
Python script to generate spatial database for PyLith based on
two velocity models from Rui. Outside of this region, results are
merged into PREM.
"""

import numpy as np
import pdb
import scipy.interpolate
from pyproj import Transformer
import meshio
import platform
# For now, if we are running Python 2, we will also assume PyLith 2.
PYTHON_MAJOR_VERSION = int(platform.python_version_tuple()[0])

if (PYTHON_MAJOR_VERSION == 2):
    from spatialdata.geocoords.CSGeoProj import CSGeoProj
    from spatialdata.geocoords.Converter import convert
    from spatialdata.spatialdb.SimpleGridAscii import SimpleGridAscii
    from coordsys_v2 import cs_geo
    from coordsys_v2 import cs_meshDaliangshan
    outSpatialDb = 'merged_velmodel_v2.spatialdb'
else:
    from spatialdata.geocoords.CSGeo import CSGeo
    from spatialdata.geocoords.CSGeoLocal import CSGeoLocal
    from spatialdata.geocoords.Converter import convert
    from spatialdata.spatialdb.SimpleGridAscii import createWriter
    from coordsys_v3 import cs_geo
    from coordsys_v3 import cs_meshDaliangshan
    outSpatialDb = 'merged_velmodel_v3.spatialdb'

# pdb.set_trace()


# Filenames.
velModelFile1 = 'liu_etal_2021_SRL_0.5x0.5_wrt_surface_VpVs.txt'
velModelFile2 = 'vps_3d_tomodd_mod_wujinaping.dat'
premFile = 'prem_1s.csv'
outOrig1VTK = 'original_velmodel_medres.vtk'
outOrig2VTK = 'original_velmodel_highres.vtk'
outPremVTK = 'prem_velmodel.vtk'
outMergedVTK = 'merged_velmodel.vtk'

# Mesh boundaries.
meshLonMin = 90.0
meshLonMax = 114.0
meshLatMin = 16.0
meshLatMax = 39.0
meshElevMin = -402000.0
meshElevAdd = [2000.0, 4000.0, 6000.0]

# Velocity model dimensions.
num1VX = 23
num1VY = 27
# Note that this includes 3 extra surface layers above z=0.
num1VZ = 15
num1V = num1VX*num1VY*num1VZ

# Amount to extend each direction.
numAddHoriz = 10
numAddZMin = 8

# Interpolation distance in each direction.
numInterpHoriz = 3
numInterpVert = 3

# Projections.
WGS84 = "+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0"
TM = "+proj=tmerc +lon_0=102.5 +lat_0=28.0 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0"
transWGS84ToTM = Transformer.from_crs(WGS84, TM, always_xy=True)

# Coordinate systems.
csGeo = cs_geo()

#-------------------------------------------------------------------------------
def getVel2Line(lines, startInd, numVals):
    """
    Get line from high resolution velocity model file.
    """
    totVals = 0
    nextInd = startInd
    vals = []
    while (totVals < numVals):
        line = lines[nextInd].split()
        nextInd += 1
        lineVals = [float(i) for i in line]
        vals += lineVals
        totVals = len(vals)

    return (vals, nextInd)


def readVelMod2(inFile):
    """
    Read higer resolution velocity model.
    """
    f = open(inFile, 'r')
    lines = f.readlines()
    line0Split = lines[0].split()
    numLat = int(line0Split[0])
    numLon = int(line0Split[1])
    numDepth = int(line0Split[2])
    numPoints = numLat*numLon*numDepth

    (lats, nextInd) = getVel2Line(lines, 1, numLat)
    (lons, nextInd) = getVel2Line(lines, nextInd, numLon)
    (depths, nextInd) = getVel2Line(lines, nextInd, numDepth)
    depths.reverse()
    lats = np.array(lats)
    lons = np.array(lons)
    depths = -1000.0*np.array(depths)

    coordsGeog = np.zeros((numPoints, 3), dtype=np.float64)

    vp = np.zeros(numPoints, dtype=np.float64)
    vs = np.zeros(numPoints, dtype=np.float64)
    for i in range(numLat):
        for j in range(numLon):
            (vpLine, nextInd) = getVel2Line(lines, nextInd, numDepth)
            vpLine.reverse()
            ind = numDepth*(i*numLon + j)
            vp[ind:ind+numDepth] = vpLine
            coordsGeog[ind:ind+numDepth,0] = lons[j]
            coordsGeog[ind:ind+numDepth,1] = lats[i]
            coordsGeog[ind:ind+numDepth,2] = depths

    for i in range(numLat):
        for j in range(numLon):
            (vsLine, nextInd) = getVel2Line(lines, nextInd, numDepth)
            vsLine.reverse()
            ind = numDepth*(i*numLon + j)
            vs[ind:ind+numDepth] = vsLine

    # Sort by lon, lat, depth.
    sortInds = np.lexsort((coordsGeog[:,0], coordsGeog[:,1], coordsGeog[:,2]))
    coordsGeog = coordsGeog[sortInds,:]
    vp = vp[sortInds]
    vs = vs[sortInds]
            
    return (coordsGeog, vp, vs, lons, lats, depths)
            
    
def brocherDensity(vp):
    """
    Function for Brocher (2005) density/Vp relation.
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
    vrange = np.geomspace(refVal, np.abs(dv) + refVal, num=numAdd)
    # vrange = np.geomspace(v1, v2, num=numAdd)
    if (dv < 0.0):
        vLRange = np.flipud(v1 + refVal - vrange)
        # vLRange = np.flipud(vrange)
    else:
        vLRange = v1 - refVal + vrange
        # vLRange = vrange

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

    connectArr = np.array(connect, dtype=np.int64)
    cellTypes = 12*np.ones(numCells, dtype=np.int64)

    # VTK header.
    vtkHead = "# vtk DataFile Version 2.0\n" + \
        "Velocity model\n" + \
        "ASCII\n" + \
        "DATASET UNSTRUCTURED_GRID\n" + \
        "POINTS %d double\n" % numPoints
  
    # Write coordinates.
    v = open(file, 'w')
    v.write(vtkHead)
    np.savetxt(v, coords, fmt="%15.11e", delimiter='\t')

    # Write connectivity.
    numCellEntries = numCells*9
    v.write("CELLS %d %d\n" % (numCells, numCellEntries))
    np.savetxt(v, connectArr, fmt="%d")
    v.write("CELL_TYPES %d\n" % numCells)
    np.savetxt(v, cellTypes, fmt="%d")

    # Write data fields.
    v.write("POINT_DATA %d\n" % numPoints)
    v.write("SCALARS vp double 1\n")
    v.write("LOOKUP_TABLE default\n")
    np.savetxt(v, vp)
    v.write("SCALARS vs double 1\n")
    v.write("LOOKUP_TABLE default\n")
    np.savetxt(v, vs)
    v.write("SCALARS density double 1\n")
    v.write("LOOKUP_TABLE default\n")
    np.savetxt(v, density)
    v.write("SCALARS mesh_part double 1\n")
    v.write("LOOKUP_TABLE default\n")
    np.savetxt(v, meshPart)

    v.close()

    return


#-------------------------------------------------------------------------------
# Read high resolution velocity model file.
print("Reading high resolution velocity model:")
(vCoordsGeog2, vVp2, vVs2, vLons2, vLats2, vElevs2) = readVelMod2(velModelFile2)
vDensity2 = brocherDensity(vVp2)
(xCart2, yCart2, zCart2) = transWGS84ToTM.transform(vCoordsGeog2[:,0], vCoordsGeog2[:,1], vCoordsGeog2[:,2])
coordsCart2 = np.column_stack((xCart2, yCart2, zCart2))
num2VX = vLons2.shape[0]
num2VY = vLats2.shape[0]
num2VZ = vElevs2.shape[0]

print("Reading original velocity model:")
vDat = np.loadtxt(velModelFile1, dtype=np.float64)
vDat[:,2] *= -1000.0

# Create fake layers above z=0.
vDatTop = vDat[:num1VX*num1VY,:]
vDatExt = vDat.copy()
for layer in range(3):
    vDatExt = np.vstack((vDatTop[:num1VX*num1VY,:], vDatExt))
    vDatExt[:num1VX*num1VY, 2] = meshElevAdd[layer]
    
# Reorder array so z goes from min to max.
vDatFlipped = np.zeros_like(vDatExt)
sliceSize = num1VX*num1VY
for sliceNum in range(num1VZ):
    indOrig = (num1VZ - sliceNum - 1)*sliceSize
    indFlipped = sliceNum*sliceSize
    vDatFlipped[indFlipped:indFlipped+sliceSize] = vDatExt[indOrig:indOrig+sliceSize]

vLons1 = vDatFlipped[0:num1VX,0]
vLats1 = vDatFlipped[0:num1VX*num1VY:num1VX,1]
vElevs1 = vDatFlipped[0::num1VX*num1VY,2]
vCoordsGeog1 = vDatFlipped[:,0:3]
vVp1 = vDatFlipped[:,3]
vVs1 = vDatFlipped[:,4]
vDensity1 = brocherDensity(vVp1)

(xCart1, yCart1, zCart1) = transWGS84ToTM.transform(vCoordsGeog1[:,0], vCoordsGeog1[:,1], vCoordsGeog1[:,2])
coordsCart1 = np.column_stack((xCart1, yCart1, zCart1))

# Get indices of old velocity model to use for interpolation.
print("Getting interpolation indices:")
xInner1 = np.logical_and(vCoordsGeog1[:,0] >= vLons2[0], vCoordsGeog1[:,0] <= vLons2[-1])
yInner1 = np.logical_and(vCoordsGeog1[:,1] >= vLats2[0], vCoordsGeog1[:,1] <= vLats2[-1])
zInner1 = np.logical_and(vCoordsGeog1[:,2] >= vElevs2[0], vCoordsGeog1[:,2] <= vElevs2[-1])
indsInside1 = xInner1*yInner1*zInner1
indsOutside1 = np.logical_not(indsInside1)
coordsUse1 = coordsCart1[indsOutside1,:]
vVpUse1 = vVp1[indsOutside1]
vVsUse1 = vVs1[indsOutside1]
vDensityUse1 = vDensity1[indsOutside1]

# Get new mesh ranges.
print("Creating new mesh:")
dLon = vLons2[1] - vLons2[0]
dLat = vLats2[1] - vLats2[0]
dElev = vElevs2[1] - vElevs2[0]
numLonsInner = int((vLons1[-1] - vLons1[0])/dLon) + 1
lonsInner = np.linspace(vLons1[0], vLons1[-1], num=numLonsInner, dtype=np.float64)
numLatsInner = int((vLats1[-1] - vLats1[0])/dLat) + 1
latsInner = np.linspace(vLats1[0], vLats1[-1], num=numLatsInner, dtype=np.float64)
numElevsInner = int((vElevs1[-1] - vElevs1[0])/dElev) + 1
elevsInner = np.linspace(vElevs1[0], vElevs1[-1], num=numElevsInner, dtype=np.float64)

westLonsAdd = createLogRange(vLons1[0], meshLonMin, numAddHoriz + 1, 10.0)
eastLonsAdd = createLogRange(vLons1[-1], meshLonMax, numAddHoriz + 1, 10.0)
southLatsAdd = createLogRange(vLats1[0], meshLatMin, numAddHoriz + 1, 10.0)
northLatsAdd = createLogRange(vLats1[-1], meshLatMax, numAddHoriz + 1, 10.0)
bottomElevsAdd = createLogRange(vElevs1[0], meshElevMin, numAddZMin + 1, 10000.0)

# New ranges.
lonsNew = np.concatenate((westLonsAdd, lonsInner[1:-1], eastLonsAdd))
latsNew = np.concatenate((southLatsAdd, latsInner[1:-1], northLatsAdd))
elevsNew = np.concatenate((bottomElevsAdd, elevsInner[1:]))

# Extended dimensions.
numVXExt = lonsNew.shape[0]
numVYExt = latsNew.shape[0]
numVZExt = elevsNew.shape[0]
numExt = numVXExt*numVYExt*numVZExt

# New grid.
(zz, yy, xx) = np.meshgrid(elevsNew, latsNew, lonsNew, indexing='ij')
coordsNewGeog = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
(xNewCart, yNewCart, zNewCart) = transWGS84ToTM.transform(xx.flatten(), yy.flatten(), zz.flatten())
coordsNewCart = np.column_stack((xNewCart, yNewCart, zNewCart))

# PREM.
print("Reading PREM velocity model:")
pDat = np.loadtxt(premFile, delimiter=',', dtype=np.float64)
numP = pDat.shape[0] + 3
pElevs = np.zeros(numP, dtype=np.float64)
pVp = np.zeros(numP, dtype=np.float64)
pVs = np.zeros(numP, dtype=np.float64)
pDensity = np.zeros(numP, dtype=np.float64)
pElevs[3:] = -1000.0*pDat[:,1]
pDensity[3:] = pDat[:,2]
pVp[3:] = 0.5*(pDat[:,3] + pDat[:,4])
pVs[3:] = 0.5*(pDat[:,5] + pDat[:,6])

# Correct near-surface for PREM.
pElevs[0] = meshElevAdd[-1]
pElevs[1] = meshElevAdd[-2]
pElevs[2] = meshElevAdd[-3]
pDensity[0:7] = pDensity[7]
pVp[0:7] = pVp[7]
pVs[0:7] = pVs[7]

# Create interpolation functions for PREM.
pVpf = scipy.interpolate.interp1d(pElevs, pVp)
pVsf = scipy.interpolate.interp1d(pElevs, pVs)
pDensityf = scipy.interpolate.interp1d(pElevs, pDensity)

print("Computing logical arrays for interpolation:")
# Define indices for original velocity model, PREM, and interpolated values.
xOrig = np.logical_and(coordsNewGeog[:,0] >= vLons1[0], coordsNewGeog[:,0] <= vLons1[-1])
yOrig = np.logical_and(coordsNewGeog[:,1] >= vLats1[0], coordsNewGeog[:,1] <= vLats1[-1])
zOrig = coordsNewGeog[:,2] >= vElevs1[0]
indsOrig = xOrig*yOrig*zOrig
coordsOrig = coordsNewCart[indsOrig,:]

# Define indices for interpolated region.
numOuterHoriz = numAddHoriz - numInterpHoriz
numOuterVert = numAddZMin - numInterpVert
xInner = np.logical_and(coordsNewGeog[:,0] >= lonsNew[numOuterHoriz], coordsNewGeog[:,0] <= lonsNew[-numOuterHoriz-1])
yInner = np.logical_and(coordsNewGeog[:,1] >= latsNew[numOuterHoriz], coordsNewGeog[:,1] <= latsNew[-numOuterHoriz-1])
zInner = coordsNewGeog[:,2] >= elevsNew[numOuterVert]
indsInner = xInner*yInner*zInner
indsInterp = np.logical_xor(indsOrig, indsInner)
coordsInterp = coordsNewCart[indsInterp,:]

# Define indices of outer parts of velocity model.
indsOuter = np.logical_xor(indsInner, np.ones_like(indsInner))
coordsOuter = coordsNewCart[indsOuter,:]
vpOuter = pVpf(coordsOuter[:,2])
vsOuter = pVsf(coordsOuter[:,2])
densityOuter = pDensityf(coordsOuter[:,2])

# Create interpolation functions.
refCoords = np.vstack((coordsCart2, coordsUse1, coordsOuter))
refVp = np.concatenate((vVp2, vVpUse1, vpOuter))
refVs = np.concatenate((vVs2, vVsUse1, vsOuter))
refDensity = np.concatenate((vDensity2, vDensityUse1, densityOuter))

print("Computing interpolation functions:")
interpVp = scipy.interpolate.LinearNDInterpolator(refCoords, refVp)
interpVs = scipy.interpolate.LinearNDInterpolator(refCoords, refVs)
interpDensity = scipy.interpolate.LinearNDInterpolator(refCoords, refDensity)

# Compute interpolated values.
print("Applying interpolation functions:")
print("Interpolating Vp:")
vpInterp = interpVp(coordsNewCart)
print("Interpolating Vs:")
vsInterp = interpVs(coordsNewCart)
print("Interpolating density:")
densityInterp = interpDensity(coordsNewCart)

# Write velocity model to spatialdb.
print("Writing spatial database:")
if (PYTHON_MAJOR_VERSION == 2):
    writer = SimpleGridAscii()
    writer.inventory.filename = outSpatialDb
    writer._configure()
else:
    writer = createWriter(outSpatialDb)
values = [{'name': "vp",
           'units': "km/s",
           'data': vpInterp},
          {'name': "vs",
           'units': "km/s",
           'data': vsInterp},
          {'name': "density",
           'units': "g/cm**3",
           'data': densityInterp}]
writer.write({'points': coordsNewGeog,
              'x': lonsNew,
              'y': latsNew,
              'z': elevsNew,
              'coordsys': csGeo,
              'data_dim': 3,
              'values': values})

# Write VTK files of different velocity models.
print("Writing VTK files:")
meshPart = np.ones_like(vVp1)
writeVtk(outOrig1VTK, coordsCart1, num1VX, num1VY, num1VZ, vVp1, vVs1, vDensity1, meshPart)

meshPart = np.ones_like(vVp2)
writeVtk(outOrig2VTK, coordsCart2, num2VX, num2VY, num2VZ, vVp2, vVs2, vDensity2, meshPart)

meshPart = np.zeros_like(vpInterp)
meshPart[indsOrig] = 1.0
meshPart[indsOuter] = 2.0
meshPart[indsInterp] = 3.0
writeVtk(outMergedVTK, coordsNewCart, numVXExt, numVYExt, numVZExt, vpInterp, vsInterp, densityInterp, meshPart)

vpPrem = pVpf(coordsNewCart[:,2])
vsPrem = pVsf(coordsNewCart[:,2])
densityPrem = pDensityf(coordsNewCart[:,2])
writeVtk(outPremVTK, coordsNewCart, numVXExt, numVYExt, numVZExt, vpPrem, vsPrem, densityPrem, meshPart)
