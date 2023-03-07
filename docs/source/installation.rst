.. _installation:

Installation
============

Requirements
------------

* botorch
* gpytorch
* map4
* matplotlib
* mhfp
* numpy
* pandas
* python (>= 3.7)
* ray
* rdkit
* seaborn
* scikit-learn
* scipy 
* torch

Instructions
------------

I highly recommand you to install Miniconda (https://docs.conda.io/en/latest/miniconda.html) if you want a clean python environnment. To install everything properly with `conda`, you just have to do this:

.. code-block:: bash

    conda create -n mobius -c conda-forge python=3 mkl numpy scipy pandas matplotlib rdkit seaborn sklearn torch botorch gpytorch sphinx sphinx_rtd_theme
    conda activate mobius
    pip install ray mhfp git+https://github.com/reymond-group/map4@v1.0 # To install ray, mhfp and map4 packages

We can now install the `mobius` package

.. code-block:: bash

    git clone https://git.scicore.unibas.ch/schwede/mobius.git
    cd mobius
    pip install -e .
