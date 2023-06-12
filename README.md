# Mobius

A python package for optimizing peptide sequences using Bayesian optimization (BO)

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
from mobius import Planner, SequenceGA
from mobius import Map4Fingerprint
from mobius import GPModel, ExpectedImprovement, TanimotoSimilarityKernel
from mobius import LinearPeptideEmulator
from mobius import homolog_scanning, alanine_scanning
from mobius import convert_FASTA_to_HELM
```

Simple linear peptide emulator/oracle for MHC class I A*0201. The Position Specific Scoring Matrices
(PSSM) can be downloaded from the [IEDB](http://tools.iedb.org/mhci/download/) database (see `Scoring 
matrices of SMM and SMMPMBEC` section). WARNING: This is for benchmarking purpose only. This step should be an 
actual lab experiment.
```python
pssm_files = ['IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-8.txt',
              'IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-9.txt',
              'IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-10.txt',
              'IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-11.txt']
lpe = LinearPeptideEmulator(pssm_files)
```

Now we define a peptide sequence we want to optimize
```python
lead_peptide = convert_FASTA_to_HELM('HMTEVVRRC')[0]
```

Then we generate the first seed library of 96 peptides using a combination of both alanine scanning 
and homolog scanning sequence-based strategies
```python
seed_library = [lead_peptide]

for seq in alanine_scanning(lead_peptide):
    seed_library.append(seq)
    
for seq in homolog_scanning(lead_peptide):
    seed_library.append(seq)

    if len(seed_library) >= 96:
        print('Reach max. number of peptides allowed.')
        break
```

The seed library is then virtually tested (Make/Test) using the linear peptide emulator we defined earlier.
WARNING: This is for benchmarking purpose only. This step is supposed to be an actual lab experiment.
```python
pic50_seed_library = lpe.predict(seed_library)
```

Once we got results from our first lab experiment we can now start the Bayesian Optimization (BO) First, 
we define the molecular fingerprint we want to use as well as the surrogate model (Gaussian Process),  
the acquisition function (Expected Improvement) and the optimization methode (SequenceGA).
```python
map4 = Map4Fingerprint(input_type='helm_rdkit', dimensions=4096, radius=1)
gpmodel = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
acq = ExpectedImprovement(gpmodel, maximize=False)
optimizer = SequenceGA(total_attempts=5)
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

Once acquisition function / surrogate model are defined and the parameters set in the YAML 
configuration file, we can initiate the planner method.
```python
ps = Planner(acq, optimizer, design_protocol='design_protocol.yaml')
```

Now it is time to run the optimization!!

```python
peptides = list(seed_library)[:]
pic50_scores = list(pic50_seed_library)[:]

# Here we are going to do 3 DMT cycles
for i in range(3):
    # Run optimization, recommand 96 new peptides based on existing data
    suggested_peptides, _ = ps.recommand(peptides, pic50_scores, batch_size=96)

    # Here you can add whatever methods you want to further filter out peptides
    
    # Get the pIC50 (Make/Test) of all the suggested peptides using the MHC emulator
    # WARNING: This is for benchmarking purpose only. This 
    # step is supposed to be an actual lab experiment.
    pic50_suggested_peptides = lpe.predict(suggested_peptides)
    
    # Add all the new data
    peptides.extend(list(suggested_peptides))
    pic50_scores.extend(list(pic50_suggested_peptides))
    
    best_seq = peptides[np.argmin(pic50_scores)]
    best_pic50 = np.min(pic50_scores)
    print('Best peptide found so far: %s / %.3f' % (best_seq, best_pic50))
    print('')
```

## Documentation

The installation instructions, documentation and tutorials can be found on [readthedocs.org](https://mobius.readthedocs.io/en/latest/).

## Citation

* [J. Eberhardt, M. Lill, T. Schwede. (2023). Combining Bayesian optimization with sequence- or structure-based strategies for optimization of peptide-binding protein.](https://doi.org/10.26434/chemrxiv-2023-b7l81)
