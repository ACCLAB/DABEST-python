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


if __name__ == "__main__":
    setup(
        name='dabest',
        author='Joses W. Ho',
        author_email='joseshowh@gmail.com',
        maintainer='Joses W. Ho',
        maintainer_email='joseshowh@gmail.com',
        version='0.3.9999',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy~=1.19',
            'scipy~=1.5',
            'pandas~=1.1',

            'matplotlib~=3.3',
            'seaborn~=0.11',
            'lqrt~=0.3'
        ],
        extras_require={'dev': ['pytest~=6.1', 'pytest-mpl~=0.11']},
        python_requires='~=3.6',
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Education",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Visualization",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
        url='https://acclab.github.io/DABEST-python-docs',
        download_url='https://www.github.com/ACCLAB/DABEST-python',
        license='BSD 3-clause Clear License'
    )
