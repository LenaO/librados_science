from setuptools import setup, find_packages

setup(name="radosObject",
      version="1.0",
      packages=["radosObject"],
      install_requires=['numpy', 'conduit', 'scirados'])
