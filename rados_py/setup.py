# Largely taken from
# https://blog.kevin-brown.com/programming/2014/09/24/combining-autotools-and-setuptools.html
import os, sys, os.path
from setuptools.command.egg_info import egg_info
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


class EggInfoCommand(egg_info):
    def finalize_options(self):
        egg_info.finalize_options(self)
        if "build" in self.distribution.command_obj:
            build_command = self.distribution.command_obj["build"]
            self.egg_base = build_command.build_base
            self.egg_info = os.path.join(self.egg_base, os.path.basename(self.egg_info))

setup(
    name = 'radoscache',
    version = '0.1',
    description = "Python libraries for the Rados Images Files",
    long_description = (
        "This package contains Python libraries for interacting with the Ceph -Rados Object interface"),
    ext_modules = cythonize([
        Extension("radoscache",
           sources = ['../src/rados_cache.cpp'],
           libraries=["rados", "radosdataset"],
           extra_compile_args=['-std=c++11',"-O3","-DHAVE_NUMPY"],
            include_dirs=['/gpfs/homeb/pcp0/pcp0063/ceph_cache/include' ],
            library_dirs=['/gpfs/homeb/pcp0/pcp0063/ceph_cache/lib']
            )
    ], build_dir=os.environ.get("CYTHON_BUILD_DIR", None)),
    cmdclass={
        "egg_info": EggInfoCommand,
    },

)


