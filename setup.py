import setuptools
from os import path
from distutils.command.build_ext import build_ext as DistUtilsBuildExt
from setuptools import setup, find_packages

VERSION = '0.2.9'


# set a long description which is basically the README
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()


setup(
    name='ark-analysis',
    version=VERSION,
    packages=find_packages(),
    license='Modified Apache License 2.0',
    description='Toolbox for analysis on segmented images from MIBI',
    author='Angelo Lab',
    url='https://github.com/angelolab/ark-analysis',
    download_url='https://github.com/angelolab/ark-analysis/archive/v{}.tar.gz'.format(VERSION),
    install_requires=['jupyter>=1.0.0,<2',
                      'jupyter_contrib_nbextensions>=0.5.1,<1',
                      'matplotlib>=2.2.2,<3',
                      'numpy>=1.16.3,<2',
                      'pandas>=0.23.3,<1',
                      'requests>=2.25.1,<3',
                      'scikit-image>=0.14.3,<=0.16.2',
                      'scikit-learn>=0.19.1,<1',
                      'scipy>=1.1.0,<2',
                      'seaborn>=0.10.1,<1',
                      'statsmodels>=0.11.1,<1',
                      'umap-learn>=0.4.6,<1',
                      'xarray>=0.12.3,<1',
                      'tqdm>=4.54.1,<5'],
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
