#!/usr/bin/env python

# Coordinate systems

from spatialdata.geocoords.CSGeo import CSGeo
from spatialdata.geocoords.CSGeoLocal import CSGeoLocal

# Geographic lat/lon coordinates in WGS84 datum
def cs_geo():
    cs = CSGeo()
    # cs.crsString = "EPSG:4326"
    cs.crsString = "+proj=lonlat +ellps=WGS84 +datum=WGS84 +towgs84=0.0,0.0,0.0 always_xy=True"
    cs.spaceDim = 3
    cs._configure()
    return cs


# Coordinate system used in Daliangshan finite-element meshes
def cs_meshDaliangshan():
    cs = CSGeo()
    cs.crsString = "+proj=tmerc +datum=WGS84 +lon_0=102.5 +lat_0=28.0 +k=0.9996 +units=m"
    cs.spaceDim = 3
    cs._configure()
    return cs


# End of file
