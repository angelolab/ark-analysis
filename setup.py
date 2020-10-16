import setuptools
from os import path
from distutils.command.build_ext import build_ext as DistUtilsBuildExt
from setuptools import setup, find_packages

VERSION = '0.3.6'
VERSION_MIBILIB = '1.3.0'


# define a parsing function for requirements.txt
def _parse_requirements(file_path):
    # people should download mibilib separately for now
    reqs = [line.strip() for line in open(file_path) if not (line.startswith('#')
            or line.startswith('git+'))]
    return reqs


# set a long description which is basically the README
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

# don't set install_reqs if we can't read requirements.txt
try:
    install_reqs = _parse_requirements('requirements.txt')
except Exception as e:
    install_reqs = []


setup(
    name='ark-analysis',
    version=VERSION,
    packages=find_packages(),
    license='Modified Apache License 2.0',
    description='Toolbox for analysis on segmented images from MIBI',
    author='Angelo Lab',
    url='https://github.com/angelolab/ark-analysis',
    download_url='https://github.com/angelolab/ark-analysis/archive/v{}.tar.gz'.format(VERSION),
    install_requires=install_reqs,
    dependency_links=['http://github.com/ionpath/mibilib/archive/v{}.zip'.format(VERSION_MIBILIB)],
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
