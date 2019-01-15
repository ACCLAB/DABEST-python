from setuptools import setup, find_packages
import os
# Taken from setup.py in seaborn.
# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"]="."

DESCRIPTION = 'Data Analysis and Visualization using Bootstrap-Coupled Estimation.'
LONG_DESCRIPTION = """\
Estimation statistics is a simple framework <https://thenewstatistics.com/itns/>
that—while avoiding the pitfalls of significance testing—uses familiar statistical
concepts: means, mean differences, and error bars. More importantly, it focuses on
the effect size of one's experiment/intervention, as opposed to
significance testing.

An estimation plot has two key features. Firstly, it presents all
datapoints as a swarmplot, which orders each point to display the
underlying distribution. Secondly, an estimation plot presents the
effect size as a bootstrap 95% confidence interval on a separate but
aligned axes.

Please cite this work as:
Moving beyond P values: Everyday data analysis with estimation plots
Joses Ho, Tayfun Tumkaya, Sameer Aryal, Hyungwon Choi, Adam Claridge-Chang
https://doi.org/10.1101/377978
"""


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
        if need_to_install(seaborn, SNS_LATEST_MAJOR, SNS_LATEST_MINOR):
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
        maintainer='Joses W. Ho',
        maintainer_email='joseshowh@gmail.com',
        version='0.1.7',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=installs,
        url='https://acclab.github.io/DABEST-python-docs',
        download_url='https://www.github.com/ACCLAB/DABEST-python',
        license='BSD 3-clause Clear License'
    )
