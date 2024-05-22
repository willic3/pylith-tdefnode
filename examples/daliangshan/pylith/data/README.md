This directory contains the site location information for this study.
To create data coordinates (and a corresponding VTK file) in the local
Transverse Mercator projection, just do:

./get_data_coords.py

This is a simple Python script that will create the .txt and .vtk files.

NOTE:
There is a new script (get_coords.py) that combines a new set of data
coordinates (contained in dls2022_all.vel) with the original
coordinates. At present, it appears that all the original coordinates
are contained in dls2022_all.vel. Note that there also appear to be
5 pairs of sites that have the same coordinates. The new script may
be run as:

./get_coords.py
