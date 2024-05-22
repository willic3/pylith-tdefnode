#!/usr/bin/env python

import numpy as np
import math
from pyproj import Transformer
import scipy.spatial.distance

# Definitions for each input file.
prefs = ["dls2023_estnoise_eura14", "dls2022_all", "GNSS_sites_available"]
suffs = [".sites", ".vel", ".txt"]
skips = [0, 0, 0]
cols = [(0,1,2), (0,1,12), (0,1,2)]
delims = [None, None, None]

# Combined results.
combinedPref = "all_sites_2023"

# Epsilon values.
uniqueEps = 0.003
coincidentEps = 0.00005

# Elevation value to assign.
zVal = -1.0

# Coordinate conversion information.
paramsWGS84 = '+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0'
paramsTM = '+proj=tmerc +lon_0=102.5 +lat_0=28.0 +ellps=WGS84 +datum=WGS84 +k=0.9996 +towgs84=0.0,0.0,0.0'

# VTK Header
vtkHeadTop = "# vtk DataFile Version 2.0\n" + \
             "Observation locations\n" + \
             "ASCII\n" + \
             "DATASET POLYDATA\n"

# Headers
WGS84Head = "#Longitude\tLatitude\tSite\n"
TMHead = "#Site\tEastingTM\tNorthingTM\tElevationTM\n"

# Define coordinate conversion.
transWGS84ToTM = Transformer.from_crs(paramsWGS84, paramsTM, always_xy=True)

# Arrays to hold unique sites.
lonsTotal = None
latsTotal = None
stnsTotal = None
numUnique = 0

# Summary info file.
fsumm = open("summary_info.txt", 'w')

def getUniqueSites(lonsList, latsList, stnsList):
    """
    Function to add unique sites to the global list.
    """
    numAdd = 0
    numInList = lonsList.shape[0]
    lonsAddList = []
    latsAddList = []
    stnsAddList = []
    for siteNum in range(numInList):
        unique = True
        stnTest = stnsList[siteNum]
        if (stnTest in stnsTotal):
            globalInd = np.where(stnsTotal == stnTest)
            lonDiff = math.fabs(lonsList[siteNum] - lonsTotal[globalInd][0])
            testLon = lonDiff > uniqueEps
            latDiff = math.fabs(latsList[siteNum] - latsTotal[globalInd][0])
            testLat = latDiff > uniqueEps
            if (testLon or testLat):
                msg = "Coordinate mismatch for site %s." % stnsList[siteNum]
                raise ValueError(msg)
            unique = False
        if (unique):
            print("  Added site %s:" % stnsList[siteNum])
            lonsAddList.append(lonsList[siteNum])
            latsAddList.append(latsList[siteNum])
            stnsAddList.append(stnsList[siteNum])

    lonsAdd = np.array(lonsAddList, dtype=np.float64)
    latsAdd = np.array(latsAddList, dtype=np.float64)
    stnsAdd = np.array(stnsAddList, dtype='|S8')
    numAdd = lonsAdd.shape[0]

    return (lonsAdd, latsAdd, stnsAdd, numAdd)


def writeCartesian(coords, sites, txtHead, fileRoot, fileCs):
    """
    Function to write both text and VTK output for a set of coordinates.
    """
    fileTxt = fileRoot + "_" + fileCs + ".txt"
    fileVtk = fileRoot + "_" + fileCs + ".vtk"
    numPoints = coords.shape[0]
    vtkHeadBot = "POINTS %d double\n" % numPoints
    outFmt = "%s\t%15.11e\t%15.11e\t%15.11e\n"

    # Write text file.
    f = open(fileTxt, 'w')
    f.write(txtHead)
    numPoints = coords.shape[0]
    for pointNum in range(numPoints):
        outLine = outFmt % (sites[pointNum], coords[pointNum,0], coords[pointNum,1], coords[pointNum,2])
        f.write(outLine)
    
    # np.savetxt(f, coords, delimiter='\t')
    f.close()

    # Write VTK file.
    f = open(fileVtk, 'w')
    f.write(vtkHeadTop)
    f.write(vtkHeadBot)
    np.savetxt(f, coords, delimiter='\t')
    f.close()

    return


def writeWGS84(lons, lats, sites, fileRoot):
    """
    Function to write text output for geographic coordinates.
    """
    fileTxt = fileRoot + "_wgs84" + ".txt"
    numPoints = lons.shape[0]

    # Write text file.
    f = open(fileTxt, 'w')
    f.write(WGS84Head)
    for pointNum in range(numPoints):
        outLine = "%10.4f\t%10.4f\t%s\n" % (lons[pointNum], lats[pointNum], sites[pointNum])
        f.write(outLine)

    f.close()

    return


def readVals(inFile, skiprows=0, usecols=[0,1,2,3,4], delimiter="\t"):
    """
    Function to read requested values from file.
    """
    f = open(inFile, 'r')
    lines = f.readlines()
    numLines = len(lines)
    numVals = numLines - skiprows
    lons = np.zeros(numVals, dtype=np.float64)
    lats = np.zeros(numVals, dtype=np.float64)
    stns = np.empty(numVals, dtype='S8')

    for lineNum in range(skiprows,numLines):
        line = lines[lineNum]
        vals = line.split(delimiter)
        lonDeg = float(vals[usecols[0]])
        lonMin = float(vals[usecols[1]])
        latDeg = float(vals[usecols[2]])
        latMin = float(vals[usecols[3]])
        stns[lineNum] = vals[usecols[4]]
        lons[lineNum] = lonDeg + lonMin/60.0
        lats[lineNum] = latDeg - latMin/60.0

    return (lons, lats, stns)


def findDuplicateSites(lons, lats, stns):
    """
    Function to find sites with duplicated coordinates.
    """
    coords = np.column_stack((lons, lats))
    distance = scipy.spatial.distance.pdist(coords)
    distMat = scipy.spatial.distance.squareform(distance)
    closePoints = np.where(distMat < coincidentEps)
    (uniq, counts) = np.unique(closePoints[0], return_counts=True)
    duplicates = np.where(counts > 1)
    numDupSites = duplicates[0].shape[0]
    print("Number of duplicated sites:  %d" % numDupSites)
    #*********NOTE:  the loop below double-counts the duplicates since
    # the distance matrix is symmetric.  See if I can fix this.
    for dupNum in range(numDupSites):
        inds = np.where(closePoints[0] == duplicates[0][dupNum])
        print("  Duplicate site:  %d" % dupNum)
        numDupsForSite = inds[0].shape[0]
        for siteDup in range(numDupsForSite):
            siteNum = closePoints[1][inds[0][siteDup]]
            site = stns[siteNum]
            print("    %s:  %f  %f" % (site, lons[siteNum], lats[siteNum]))

    return


# Read WGS84 coordinates from input files.
for fileNum in range(len(prefs)):
    inFile = prefs[fileNum] + suffs[fileNum]
    if (len(cols[fileNum]) < 5):
        (lons, lats, stns) = np.loadtxt(inFile, skiprows=skips[fileNum], usecols=cols[fileNum], unpack=True,
                                        delimiter=delims[fileNum], dtype=str)
        lons = np.float64(lons)
        lats = np.float64(lats)
    else:
        (lons, lats, stns) = readVals(inFile, skiprows=skips[fileNum], usecols=cols[fileNum], delimiter=delims[fileNum])

    negLons = np.where(lons < 0.0)
    lons[negLons] += 360.0
    numOrig = lons.shape[0]

    if (fileNum == 0):
        lonsTotal = lons.copy()
        latsTotal = lats.copy()
        stnsTotal = stns.copy()
        numUnique = lonsTotal.shape[0]
        numAdd = numUnique
    else:
        (lonsAdd, latsAdd, stnsAdd, numAdd) = getUniqueSites(lons, lats, stns)
        if (numAdd != 0):
            lonsTotal = np.append(lonsTotal, lonsAdd)
            latsTotal = np.append(latsTotal, latsAdd)
            stnsTotal = np.append(stnsTotal, stnsAdd)
            numUnique = lonsTotal.shape[0]

    # Write info to summary file.
    fsumm.write("Input file:  %s\n" % inFile)
    fsumm.write("  Number of original stations:         %d\n" % (numOrig))
    fsumm.write("  Number of unique stations added:     %d\n" % (numAdd))
    fsumm.write("\n")

    # Perform coordinate transformations and print.
    depths = zVal * np.ones_like(lons)
    (xTM, yTM, zTM) = transWGS84ToTM.transform(lons, lats, depths)
    coordsTM = np.column_stack((xTM, yTM, zTM))

    writeCartesian(coordsTM, stns, TMHead, prefs[fileNum], 'tm')
    writeWGS84(lons, lats, stns, prefs[fileNum])

# Look for duplicate sites.
findDuplicateSites(lonsTotal, latsTotal, stnsTotal)

# Finish summary file.
fsumm.write("Total number of unique sites:  %d\n" % numUnique)
fsumm.close()

# Print combined site list.
print("")
print("Number of unique sites:  %d" % numUnique)
depths = zVal * np.ones_like(lonsTotal)
(xTM, yTM, zTM) = transWGS84ToTM.transform(lonsTotal, latsTotal, depths)
coordsTM = np.column_stack((xTM, yTM, zTM))

writeCartesian(coordsTM, stnsTotal, TMHead, combinedPref, 'tm')
writeWGS84(lonsTotal, latsTotal, stnsTotal, combinedPref)

