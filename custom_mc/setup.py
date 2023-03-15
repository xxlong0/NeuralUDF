# python setup.py build_ext --inplace
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

includes_numpy = '-I ' + np.get_include() + ' '
os.environ['CFLAGS'] = includes_numpy + (os.environ['CFLAGS'] if 'CFLAGS' in os.environ else '')

setup(
    name="My MC",
    ext_modules=cythonize("_marching_cubes_lewiner_cy.pyx", include_path=[np.get_include()], language="c++"),
)
