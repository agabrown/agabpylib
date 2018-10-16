"""
Setup module for agabpylib

Anthony Brown Oct 2015 - Oct 2018

Based on:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

this_folder = path.abspath(path.dirname(__file__))

# Get the long description from the README and HISTORY files
with open(path.join(this_folder, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
with open(path.join(this_folder, 'HISTORY.rst'), encoding='utf-8') as f:
    history = f.read()

# Get code version from __init__.py (see https://github.com/dfm/emcee/blob/master/setup.py)
vre = re.compile("__version__ = \"(.*?)\"")
m = open(path.join(this_folder, "agabpylib", "__init__.py")).read()
codeVersion = vre.findall(m)[0]

setup(
    name="agabpylib",
    version=codeVersion,
    description="Python utilities written by Anthony Brown",
    long_description=long_description + "\n\n"
    + "Changelog\n"
    + "---------\n\n"
    + history,

    # The project's main homepage.
    url="https://github.com/agabrown/agabpylib",

    author="Anthony Brown",
    author_email="brown@strw.leidenuniv.nl",
    license="MIT",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 0 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT',
        'Operating System :: OS Independent',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Astronomy',
    ],

    # What does your project relate to?
    keywords='plotting, data analysis, numerical tools',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'doc', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #py_modules=["module_a", "module_b"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy','scipy', 'matplotlib', 'scikit-learn', 'pandas'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={},

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={'': ['LICENSE', 'AUTHORS.rst', 'HISTORY.rst', 'INSTALL',
        'MANIFEST.in'], 'agabpylib':[], 'agabpylib':['data/*.txt'], 'agabpylib.plotting':['data/*.rgb'],
        'agabpylib.plotting':['data/*'], 'agabpylib.simulation':['data/*'], 
        'agabpylib.gaia':['data/*']},
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={},
)

