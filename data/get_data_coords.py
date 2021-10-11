#!/usr/bin/env python

"""
Python script to get data coordinates and output them in different coordinate
systems. This version is for stations at a constant elevation.
"""

import numpy
from pyproj import Proj
from pyproj import transform

import pdb
pdb.set_trace()

# Filenames.
inFile = "GNSS_sites_available.txt"
outFileTxt = "GNSS_sites_available_tm.txt"
outFileVTK = "GNSS_sites_available_tm.vtk"

# Assume constant z-value of -10 m.
zCorr = -10.0

# Headers.
vtkHead = \
        '# vtk DataFile Version 2.0\n' + \
        'Observation locations\n' + \
        'ASCII\n' + \
        'DATASET POLYDATA\n'
txtHead = "#Site\tX\tY\tZ\n"

# Projections.
WGS84 = "+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0"
TM = "+proj=tmerc +lon_0=102.5 +lat_0=28.0 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0"

projWGS84 = Proj(WGS84)
projTM = Proj(TM)

#----------------------------------------------------------------------------------------------------
def readCoords(file):
    """
    Function to read lon and lat from a file.
    """
    f = open(file, 'r')
    lines = f.readlines()
    numLines = len(lines)

    lon = []
    lat = []
    stn = []
    numPoints = 0

    for lineNum in range(numLines):
        line = lines[lineNum]
        if not line.startswith("#"):
            lineSplit = line.split()
            lon.append(float(lineSplit[0]))
            lat.append(float(lineSplit[1]))
            stn.append(lineSplit[2])
            numPoints += 1

    lonArr = numpy.array(lon, dtype=numpy.float64)
    latArr = numpy.array(lat, dtype=numpy.float64)
    zArr = numpy.zeros_like(lonArr) + zCorr
    f.close()

    return (lonArr, latArr, zArr, stn)


def writeStns(file, coords, stn):
    """
    Function to write station output in PyLith format.
    """
    f = open(file, 'w')
    f.write(txtHead)
    numPoints = coords.shape[0]
    fmt = "%s\t%15.11e\t%15.11e\t%15.11e\n"
    for pointNum in range(numPoints):
        outLine = fmt % (stn[pointNum], coords[pointNum, 0], coords[pointNum, 1], coords[pointNum, 2])
        f.write(outLine)

    f.close()

    return


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

#----------------------------------------------------------------------------------------------------
# Read data.
(gpsLon, gpsLat, gpsZ, gpsStn) = readCoords(inFile)

numGPS = gpsLon.shape[0]

# Do projections and stack coordinates.
(gpsEastTM, gpsNorthTM, gpsElevTM) = transform(projWGS84, projTM, gpsLon, gpsLat, gpsZ)
gpsTM = numpy.column_stack((gpsEastTM, gpsNorthTM, gpsElevTM))

# Write text file.
writeStns(outFileTxt, gpsTM, gpsStn)

# Write VTK file.
writeVtk(outFileVTK, gpsTM)
