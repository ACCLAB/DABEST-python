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



def need_to_install(module, version):
    desired_major_version = int(version.split('.')[0])
    desired_minor_version = int(version.split('.')[1])

    INSTALLED_VERSION_MAJOR = int(module.__version__.split('.')[0])
    INSTALLED_VERSION_MINOR = int(module.__version__.split('.')[1])

    if INSTALLED_VERSION_MAJOR < desired_major_version:
        return True

    elif INSTALLED_VERSION_MAJOR == desired_major_version and \
         INSTALLED_VERSION_MINOR < desired_minor_version:
        return True

    else:
        return False



def check_dependencies():
    from importlib import import_module

    modules = {'numpy'      : '1.15',
               'scipy'      : '1.2',
               'statsmodels': '0.9', 
               'pandas'     : '0.24',
               'matplotlib' : '3.0',
               'seaborn'    : '0.9'}
    to_install = []

    for module, version in modules.items():
        try:
            my_module = import_module(module)

            if need_to_install(my_module, version):
                to_install.append("{}=={}".format(module, version))

        except ImportError:
            to_install.append("{}=={}".format(module, version))

    return to_install



if __name__ == "__main__":

    installs = check_dependencies()

    setup(
        name='dabest',
        author='Joses W. Ho',
        author_email='joseshowh@gmail.com',
        maintainer='Joses W. Ho',
        maintainer_email='joseshowh@gmail.com',
        version='0.2.4',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=installs,
        url='https://acclab.github.io/DABEST-python-docs',
        download_url='https://www.github.com/ACCLAB/DABEST-python',
        license='BSD 3-clause Clear License'
    )
