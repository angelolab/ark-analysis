from os import path, pardir
from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize

CYTHON_DEBUG = False

if CYTHON_DEBUG:
    from Cython.Compiler.Options import get_directive_defaults
    directive_defaults = get_directive_defaults()

    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True

CYTHON_MACROS = [('CYTHON_TRACE', '1')] if CYTHON_DEBUG else None

VERSION = '0.5.0'

PKG_FOLDER = path.abspath(path.join(__file__, pardir))

with open(path.join(PKG_FOLDER, 'requirements.txt')) as req_file:
    requirements = req_file.read().splitlines()

# set a long description which is basically the README
with open(path.join(PKG_FOLDER, 'README.md')) as f:
    long_description = f.read()

extensions = [Extension(
    name="ark.utils._bootstrapping",
    sources=[path.join(PKG_FOLDER, 'ark', 'utils', '_bootstrapping.pyx')],
    include_dirs=[np.get_include()],
    define_macros=CYTHON_MACROS
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
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
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
                 'Programming Language :: Python :: 3.8']
)
