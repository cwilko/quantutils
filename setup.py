
from setuptools import setup, find_packages

import os

#os.system('pip install git+https://github.com/locke105/pylogmet.git')

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='quantutils',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='A set of python quantative finance tools',  # Required

    install_requires=requirements,

    extras_require={
        'stats': ['scipy>=1.1.0', 'statsmodels>=0.9.0'],
        'options': ['py-vollib-vectorized<=0.1.1'],
        'plot': ['matplotlib>=2.2.2,<3.6', 'mpl_finance>=0.10.0', 'plotly>=3.1.1', 'python-highcharts @ git+https://github.com/cwilko/python-highcharts.git']
    },
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(exclude=['contrib', 'docs', 'test', 'samples', 'build'])  # Required

)
