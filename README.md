# Mobius

A python package for optimizing peptide sequences using Bayesian optimization (BO) for multiple objectives.

## Installation

I highly recommand you to install Mamba (https://github.com/conda-forge/miniforge#mambaforge) if you want a clean python environnment. To install everything properly with `mamba`, you just have to do this:

```bash
mamba env create -f environment.yaml -n mobius
mamba activate mobius
```

We can now install the `mobius` package from the PyPI index:
```bash
# This is not a mistake, the package is called moebius on PyPI
pip install moebius
```

You can also get it directly from the source code:
```bash
pip install git+https://git.scicore.unibas.ch/schwede/mobius.git@v0.3
```

## Quick tutorial

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mobius import Map4Fingerprint
from mobius import GPModel, ExpectedImprovement, TanimotoSimilarityKernel
from mobius import LinearPeptideEmulator
from mobius import homolog_scanning
from mobius import convert_FASTA_to_HELM
from mobius import MOOProblem,MOOPlanner
from mobius import visualise_3d_scatter
from mobius import optimisation_tracker
```

Simple linear peptide emulator/oracle for MHC class I A*0201. The Position Specific Scoring Matrices
(PSSM) can be downloaded from the [IEDB](http://tools.iedb.org/mhci/download/) database (see `Scoring 
matrices of SMM and SMMPMBEC` section). WARNING: This is for benchmarking purpose only. This step should be an 
actual lab experiment.
```python
pssm_file_one = ['IEDB_MHC/smmpmbec_matrix/HLA-A-02:16-9.txt']
pssm_file_two = ['IEDB_MHC/smmpmbec_matrix/HLA-A-02:11-9.txt']
pssm_file_three = ['IEDB_MHC/smmpmbec_matrix/HLA-A-02:01-9.txt']

lpe_one = LinearPeptideEmulator(pssm_file_one)
lpe_two = LinearPeptideEmulator(pssm_file_two)
lpe_three = LinearPeptideEmulator(pssm_file_three)
```

Now we define a peptide sequence we want to optimize:
```python
lead_peptide = convert_FASTA_to_HELM('HMTEVVRRC')[0]
```

Then we generate the first seed library of 96 peptides using a homolog scanning sequence-based strategy.
```python
seed_library = [lead_peptide]

for seq in homolog_scanning(lead_peptide):
    seed_library.append(seq)

    if len(seed_library) >= 96:
        print('Reach max. number of peptides allowed.')
        break
```

The seed library is then virtually tested (Make/Test) using the linear peptide emulator we defined earlier.
WARNING: This is for benchmarking purpose only. This step is supposed to be an actual lab experiment.
```python
pic50_one_seed_library = lpe_one.score(seed_library)
pic50_two_seed_library = lpe_two.score(seed_library)
pic50_three_seed_library = lpe_three.score(seed_library)

pic50_scores = np.column_stack((pic50_one_seed_library,pic50_two_seed_library,pic50_three_seed_library))
```

Once we have the results from our first lab experiment we can now start the Bayesian Optimization (BO). First, 
we define the molecular fingerprint we want to use as well as the surrogate models for each objective (Gaussian Processes) 
and the acquisition functions (Expected Improvement).
```python
map4 = Map4Fingerprint(input_type='helm_rdkit', dimensions=4096, radius=1)

gpmodel_one = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
gpmodel_two = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
gpmodel_three = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)

acq_one = ExpectedImprovement(gpmodel_one, maximize=False)
acq_two = ExpectedImprovement(gpmodel_two, maximize=False)
acq_three = ExpectedImprovement(gpmodel_three, maximize=False)

acqs = [acq_one,acq_two,acq_three]
```

... and now let's define the search protocol in a YAML configuration file (`design_protocol.yaml`) that will be used 
to optimize the peptide sequence. This YAML configuration file defines the design protocol, in which you need 
to define the peptide scaffold, linear here. Additionnaly, you can specify the sets of monomers to be used at 
specific positions during the optimization.  You can also define some filtering criteria to remove peptide sequences 
that might exhibit some problematic properties during synthesis, such as self-aggregation or solubility.

```YAML
design:
  monomers: 
    default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
    APOLAR: [A, F, G, I, L, P, V, W]
    POLAR: [C, D, E, H, K, N, Q, R, K, S, T, M]
    AROMATIC: [F, H, W, Y]
    POS_CHARGED: [K, R]
    NEG_CHARGED: [D, E]
  scaffolds:
    - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0:
        PEPTIDE1:
          1: [AROMATIC, NEG_CHARGED]
          4: POLAR
          9: [A, V, I, L, M, T]
filters:
  - class_path: mobius.PeptideSelfAggregationFilter
  - class_path: mobius.PeptideSolubilityFilter
    init_args:
      hydrophobe_ratio: 0.5
      charged_per_amino_acids: 5

```

Once acquisition functions / surrogate models are defined and the parameters set in the YAML 
configuration file, we can initiate the multi-objective problem we are optimising for and the planner method.
```python
problem = MOOProblem(acqs)
planner = MOOPlanner(problem,design_protocol='design_protocol.yaml',batch_size=96)
```

Now it is time to run the optimization!!

```python
peptides = list(seed_library)[:]

# Initialise the DataFrame which records the progression of the optimisation
optimisation = 0
pic50_history = pd.DataFrame()
pic50_history = optimisation_tracker(optimisation,pic50_history,peptides,pic50_scores)

# Here we are going to do 3 DMT cycles
for i in range(3):

    optimisation += 1
    
    # Run optimization, recommand 96 new peptides based on existing data
    suggested_peptides_df, _ = planner.recommand(peptides, pic50_scores)
    suggested_peptides = suggested_peptides_df.iloc[:, 0].tolist()

    # Here you can add whatever methods you want to further filter out peptides
    
    # Get the pIC50 (Make/Test) of all the suggested peptides using the MHC emulator
    # WARNING: This is for benchmarking purpose only. This 
    # step is supposed to be an actual lab experiment.
    pic50_1_suggested_peptides = lpe_one.score(suggested_peptides)
    pic50_2_suggested_peptides = lpe_two.score(suggested_peptides)
    pic50_3_suggested_peptides = lpe_three.score(suggested_peptides)

    pic50_suggested_peptides = np.column_stack((pic50_1_suggested_peptides,pic50_2_suggested_peptides,pic50_3_suggested_peptides))
    
    # Add all the new data
    peptides.extend(list(suggested_peptides))
    pic50_scores = np.concatenate((pic50_scores,pic50_suggested_peptides),axis=0)

    # Update the optimisation tracker
    pic50_history = optimisation_tracker(optimisation,pic50_history,suggested_peptides,pic50_suggested_peptides)
```

Now we can also visualise the progression of the optimisation:
```python
fig,_ = visualise_3d_scatter(pic50_history)
fig.show()
```

## Documentation

The installation instructions, documentation and tutorials can be found on [readthedocs.org](https://mobius.readthedocs.io/en/latest/).

## Citation

* [J. Eberhardt, M. Lill, T. Schwede. (2023). Combining Bayesian optimization with sequence- or structure-based strategies for optimization of peptide-binding protein.](https://doi.org/10.26434/chemrxiv-2023-b7l81)
