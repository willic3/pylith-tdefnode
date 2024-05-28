from setuptools import setup, find_packages

setup(name='pylith_tdefnode',
      version='0.1',
      description='Tools for using PyLith FEM code with TDefnode slip inversion code',
      author='Charles Williams',
      author_email='c.williams@gns.cri.nz',
      packages=find_packages(),
      scripts=['pylith_tdefnode/cli/pytdef_compare_def_gf',
               'pylith_tdefnode/cli/pytdef_create_cell_size',
               'pylith_tdefnode/cli/pytdef_create_tdefnode_quads',
               'pylith_tdefnode/cli/pytdef_get_tdef_moment',
               'pylith_tdefnode/cli/pytdef_py2def',
               'pylith_tdefnode/cli/pytdef_read_def_gf'],

      requires=[
          'os',
          'glob',
          'platform',
          'matplotlib',
          'sys',
          'pathlib',
          'math',
          'numpy',
          'scipy',
          'shapely',
          'netCDF4',
          'h5py',
          'pyproj',
          'fortranformat',
          'pythia']
      )
