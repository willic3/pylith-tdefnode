from setuptools import setup, find_packages

setup(name='pylith_tdefnode',
      version='0.1',
      description='Tools for using PyLith FEM code with TDefnode slip inversion code',
      author='Charles Williams',
      author_email='c.williams@gns.cri.nz',
      packages=find_packages(),
      requires=[
          'os',
          'glob',
          'platform',
          'sys',
          'pathlib',
          'math',
          'matplotlib',
          'numpy',
          'scipy',
          'shapely',
          'netCDF4',
          'h5py',
          'pyproj',
          'fortranformat',
          'pythia']
      )
