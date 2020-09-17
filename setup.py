from os import path
from setuptools import setup, find_packages

VERSION = '0.2.4'

# set a long description which is basically the README
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name='ark',
    version=VERSION,
    packages=find_packages(),
    license='Modified Apache License 2.0',
    description='Toolbox for analysis on segmented images from MIBI',
    author='Angelo Lab',
    url='https://github.com/angelolab/ark-analysis',
    download_url='https://github.com/angelolab/ark-analysis/archive/{}.tar.gz'.format(VERSION),
    install_requires=['h5py',
                      'jupyter',
                      'jupyter_contrib_nbextensions',
                      'matplotlib',
                      'git+git://github.com/ionpath/mibilib@v1.2.6',
                      'netCDF4',
                      'numpy',
                      'pandas',
                      'scikit-image',
                      'sciit-learn',
                      'scipy',
                      'seaborn',
                      'statsmodels',
                      'tables',
                      'umap-learn',
                      'xarray'],
    extras_require={
        'tests': ['pytest',
                  'pytest-cov',
                  'pytest-pycodestyle']
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6']
)
