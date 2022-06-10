from os import path
from setuptools import setup, find_packages

VERSION = '0.2.10'

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
    install_requires=['cryptography>=3.4.8,<4',
                      'feather_format>=0.4.1,<1',
                      'ipympl>=0.7.0,<0.8.0',
                      'jupyter>=1.0.0,<2',
                      'jupyter_contrib_nbextensions>=0.5.1,<1',
                      'jupyterlab>=3.4.3,<4',
                      'matplotlib>=2.2.2,<3',
                      'numpy>=1.21.6,<2',
                      'palettable>=3.3.0,<4',
                      'pandas>=1.3.5,<2',
                      'requests>=2.27.1,<3',
                      'scikit-image>=0.14.3,<=0.16.2',
                      'scikit-learn>=0.24.2,<1',
                      'scipy>=1.7.3,<2',
                      'seaborn>=0.11.2,<1',
                      'setuptools==60.8.2',
                      'spatial-lda>=0.1.3,<1',
                      'statsmodels>=0.13.2,<1',
                      'umap-learn>=0.5.3,<1',
                      'xarray>=0.17.0,<1',
                      'tqdm>=4.64.0,<5'],
    # dependency_links=[
    #     'git+https://github.com/calico/spatial_lda.git@primary',
    # ],
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
                 'Programming Language :: Python :: 3.7']
)
