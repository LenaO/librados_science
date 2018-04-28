# Largely taken from
# https://blog.kevin-brown.com/programming/2014/09/24/combining-autotools-and-setuptools.html
import os, sys, os.path
from setuptools.command.egg_info import egg_info
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from setuptools import setup, find_packages

setup(
    name = 'scirados',
    version = '0.1',
    description = "Python libraries for the Rados Images Files",
    long_description = (
        "This package contains Python libraries for interacting with the Ceph -Rados Object interface"),
     ext_modules = [
        Extension("scirados",
           sources = ['../src/rados_cache.cpp', '../src/rados_data.cpp', '../src/rados_hierarchy.cpp'],
           libraries=["rados","conduit"],
           extra_compile_args=['-std=c++11',"-O3","-DHAVE_NUMPY "],
            include_dirs=['/gpfs/homeb/pcp0/pcp0063/ceph_cache/include' ],
            library_dirs=['/gpfs/homeb/pcp0/pcp0063/ceph_cache/lib']
            )
    ],
   install_requires=["numpy"],

)


