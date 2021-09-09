import setuptools
from os import path
from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize

VERSION = '0.2.9'

with open(path.join(path.dirname(__file__), 'requirements.txt')) as req_file:
    requirements = req_file.read().splitlines()

# set a long description which is basically the README
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

extensions = [Extension(
    name="ark.utils._bootstrapping",
    sources=["ark/utils/_bootstrapping/_close_cell_num_random.pyx"],
    include_dirs=[np.get_include()]
)]

setup(
    name='ark-analysis',
    version=VERSION,
    packages=find_packages(),
    license='Modified Apache License 2.0',
    description='Toolbox for analysis on segmented images from MIBI',
    author='Angelo Lab',
    url='https://github.com/angelolab/ark-analysis',
    download_url='https://github.com/angelolab/ark-analysis/archive/v{}.tar.gz'.format(VERSION),
    ext_modules=cythonize(extensions),
    install_requires=requirements,
    extras_require={
        'tests': ['pytest',
                  'pytest-cov',
                  'pytest-pycodestyle',
                  'testbook']
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6']
)
