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

I highly recommand you to install Mamba (https://github.com/conda-forge/miniforge#mambaforge) if you want 
a clean python environnment. To install everything properly with `mamba`, you just have to do this:

.. code-block:: bash

    mamba env create -f environment.yaml -n mobius
    mamba activate mobius

We can now install the `mobius` package from the PyPI index:

.. code-block:: bash

    # This is not a mistake, the package is called moebius on PyPI
    pip install moebius

You can also get it directly from the source code:

.. code-block:: bash

    pip install git+https://git.scicore.unibas.ch/schwede/mobius.git@v0.3
