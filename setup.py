from setuptools import setup, find_packages
import os
# Taken from setup.py in seaborn.
# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"]="."

# Modified from from setup.py in seaborn.
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def check_dependencies():
    to_install=[]

    try:
        import numpy
    except ImportError:
        to_install.append('numpy==1.13')
    try:
        import scipy
    except ImportError:
        to_install.append('scipy==1.0.0')
    try:
        import matplotlib
    except ImportError:
        to_install.append('matplotlib==2.1')
    try:
        import pandas
        if int(pandas.__version__.split('.')[1])<21:
            to_install.append('pandas==0.21')
    except ImportError:
        to_install.append('pandas==0.21')
    try:
        import seaborn
    except ImportError:
        to_install.append('seaborn==0.8')

    return to_install

if __name__=="__main__":

    installs=check_dependencies()
    setup(name='dabest',
    author='Joses Ho',
    author_email='joseshowh@gmail.com',
    version='0.0.3',
    description='Data Analysis and Visualization using Bootstrapped Estimation.',
    packages=find_packages(),
    install_requires=installs,
    url='http://github.com/josesho/bootstrap_contrast',
    license='BSD 3-clause "New" or "Revised" License'
    )
