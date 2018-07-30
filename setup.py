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



def need_to_install(library, desired_major_version, desired_minor_version):
    LIB_INSTALLED_VERSION = library.__version__
    LIB_INSTALLED_VERSION_MAJOR = int(LIB_INSTALLED_VERSION.split('.')[0])
    LIB_INSTALLED_VERSION_MINOR = int(LIB_INSTALLED_VERSION.split('.')[1])

    if LIB_INSTALLED_VERSION_MAJOR < desired_major_version:
        return True

    elif LIB_INSTALLED_VERSION_MINOR < desired_minor_version:
        return True

    else:
        return False



def check_dependencies():
    to_install = []


    NUMPY_LATEST_MAJOR = 1
    NUMPY_LATEST_MINOR = 15
    TO_INSTALL = 'numpy=={}.{}'.format(NUMPY_LATEST_MAJOR,
                                       NUMPY_LATEST_MINOR)
    try:
        import numpy
        if need_to_install(numpy, NUMPY_LATEST_MAJOR, NUMPY_LATEST_MINOR):
            to_install.append(TO_INSTALL)
    except ImportError:
        to_install.append(TO_INSTALL)


    SCIPY_LATEST_MAJOR = 1
    SCIPY_LATEST_MINOR = 1
    TO_INSTALL = 'scipy=={}.{}'.format(SCIPY_LATEST_MAJOR,
                                       SCIPY_LATEST_MINOR)
    try:
        import scipy
        if need_to_install(scipy, SCIPY_LATEST_MAJOR, SCIPY_LATEST_MINOR):
            to_install.append(TO_INSTALL)
    except ImportError:
        to_install.append(TO_INSTALL)

    PANDAS_LATEST_MAJOR = 0
    PANDAS_LATEST_MINOR = 23
    TO_INSTALL = 'pandas=={}.{}'.format(PANDAS_LATEST_MAJOR,
                                        PANDAS_LATEST_MINOR)
    try:
        import pandas
        if need_to_install(pandas, PANDAS_LATEST_MAJOR, PANDAS_LATEST_MINOR):
            to_install.append(TO_INSTALL)
    except ImportError:
        to_install.append(TO_INSTALL)


    MPL_LATEST_MAJOR = 2
    MPL_LATEST_MINOR = 2
    TO_INSTALL = 'matplotlib=={}.{}'.format(MPL_LATEST_MAJOR,
                                            MPL_LATEST_MINOR)
    try:
        import matplotlib as mpl
        if need_to_install(mpl, MPL_LATEST_MAJOR, MPL_LATEST_MINOR):
            to_install.append(TO_INSTALL)
    except ImportError:
        to_install.append(TO_INSTALL)


    SNS_LATEST_MAJOR = 0
    SNS_LATEST_MINOR = 9
    TO_INSTALL = 'seaborn=={}.{}'.format(SNS_LATEST_MAJOR,
                                        SNS_LATEST_MINOR)
    try:
        import seaborn
        if need_to_install(pandas, SNS_LATEST_MAJOR, SNS_LATEST_MINOR):
            to_install.append(TO_INSTALL)
    except ImportError:
        to_install.append(TO_INSTALL)

    return to_install



if __name__ == "__main__":

    installs = check_dependencies()

    setup(
        name='dabest',
        author='Joses W. Ho',
        author_email='joseshowh@gmail.com',
        version='0.1.4',
        description='Data Analysis and Visualization using Bootstrap-Coupled Estimation.',
        packages=find_packages(),
        install_requires=installs,
        url='https://acclab.github.io/DABEST-python-docs/index.html',
        license='BSD 3-clause Clear License'
    )
