#!/usr/bin/env python

# Coordinate systems

from spatialdata.geocoords.CSGeo import CSGeo
from spatialdata.geocoords.CSGeoProj import CSGeoProj

# Geographic lat/lon coordinates in WGS84 datum
def cs_geo():
    cs = CSGeo()
    cs.inventory.datumHoriz = "WGS84"
    cs.inventory.datumVert = "mean sea level"
    cs.inventory.spaceDim = 3
    cs._configure()
    cs.initialize()
    return cs


# Coordinate system used in Daliangshan finite-element meshes
def cs_meshDaliangshan():
    cs = CSGeoProj()
    cs.inventory.datumHoriz = "WGS84"
    cs.inventory.datumVert = "mean sea level"
    cs.inventory.spaceDim = 3
    cs.inventory.projector.inventory.projection = "tmerc"
    cs.inventory.projector.inventory.projOptions = "+lon_0=102.5 +lat_0=28.0 +k=0.9996"
    cs.inventory.projector._configure()
    cs._configure()
    cs.initialize()
    return cs


# End of file
