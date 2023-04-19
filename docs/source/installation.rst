.. _installation:

Installation
============

Requirements
------------

Mobius requires a number of Python packages, notably:

* botorch
* gpytorch
* matplotlib
* mhfp
* numpy
* pandas
* ray
* rdkit
* scikit-learn
* scipy 
* torch

Instructions
------------

I highly recommand you to install Miniconda (https://docs.conda.io/en/latest/miniconda.html) if you want a 
clean python environnment. To install everything properly with `conda`, you just have to do this:

.. code-block:: bash

    conda create -n mobius -c conda-forge python=3 mkl numpy scipy pandas matplotlib \
        rdkit seaborn sklearn torch botorch gpytorch sphinx sphinx_rtd_theme
    conda activate mobius
    pip install ray mhfp

We can now install the `mobius` package

.. code-block:: bash

    git clone https://git.scicore.unibas.ch/schwede/mobius.git
    cd mobius
    pip install -e .
