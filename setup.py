import setuptools
from os import path
from distutils.command.build_ext import build_ext as DistUtilsBuildExt
from setuptools import setup, find_packages

VERSION = '0.3.6'


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
except Exception:
    install_reqs = []


# this mess is needed for numpy to build properly and to install the requirements
class BuildExtension(setuptools.Command):
    description = DistUtilsBuildExt.description
    user_options = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


setup(
    name='ark-analysis',
    version=VERSION,
    packages=find_packages(),
    license='Modified Apache License 2.0',
    description='Toolbox for analysis on segmented images from MIBI',
    author='Angelo Lab',
    url='https://github.com/angelolab/ark-analysis',
    download_url='https://github.com/angelolab/ark-analysis/archive/v{}.tar.gz'.format(VERSION),
    cmdclass={'build_ext': BuildExtension},
    install_requires=install_reqs,
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
