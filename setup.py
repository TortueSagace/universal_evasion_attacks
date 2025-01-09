# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="versatile_evasion_attacks",
    version="1.1.4",
    # Tells setuptools to include your data files in the wheel/sdist
    include_package_data=True,  
    package_data={
        "vea": ["master/master_params.json"],  # <== Put master_params.json under the `vea` package
    },
    description="Security protocols for estimating adversarial robustness of machine learning models for both tabular and image datasets."
    + " This package implements a set of evasion attacks based on heuristic optimization algorithms, and "
    + "complex cost functions to give reliable results for tabular problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TortueSagace/versatile_evasion_attacks",
    author="Alexandre Le Mercier",
    author_email="alexandre.le.mercier@ulb.be",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(include=["vea", "vea.*"]),
    install_requires=["numpy", "matplotlib", "tqdm", "scikit-learn", "xgboost", "lightgbm",
                       "pandas", "seaborn", "optuna"],
)

"""
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
"""