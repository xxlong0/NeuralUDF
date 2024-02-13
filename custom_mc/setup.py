# python setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name="My MC",
    ext_modules=cythonize(
        Extension(
            "_marching_cubes_lewiner_cy",
            sources=["_marching_cubes_lewiner_cy.pyx"],
            include_dirs=[np.get_include()],
            language="c++"
        )
    ),
    install_requires=["numpy"]
)
