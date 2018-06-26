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
        if int(numpy.__version__.split('.')[1])<=12:
            to_install.append('numpy==1.13')
    except ImportError:
        to_install.append('numpy==1.13')

    try:
        import scipy
        if int(scipy.__version__.split('.')[0])==0:
            to_install.append('scipy==1.0')
    except ImportError:
        to_install.append('scipy==1.0')

    try:
        import pandas
        if int(pandas.__version__.split('.')[1])<=23:
            to_install.append('pandas==0.23')
    except ImportError:
        to_install.append('pandas==0.23')

    try:
        import seaborn
        if int(seaborn.__version__.split('.')[1])<=7:
            to_install.append('seaborn==0.8')
    except ImportError:
        to_install.append('seaborn==0.8')

    return to_install

if __name__=="__main__":

    installs=check_dependencies()
    setup(name='dabest',
    author='Joses W. Ho',
    author_email='joseshowh@gmail.com',
    version='0.1.4',
    description='Data Analysis and Visualization using Bootstrapped Estimation.',
    packages=find_packages(),
    install_requires=installs,
    url='https://acclab.github.io/DABEST-python-docs/index.html',
    license='BSD 3-clause Clear License'
    )
