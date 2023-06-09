# Mobius

A python package for optimizing peptide sequences using Bayesian optimization (BO)

## Installation

I highly recommand you to install Mamba (https://github.com/conda-forge/miniforge#mambaforge) if you want a clean python environnment. To install everything properly with `mamba`, you just have to do this:

```bash
mamba env create -f environment.yaml -n mobius
mamba activate mobius
```

We can now install the `mobius` package
```bash
git clone https://git.scicore.unibas.ch/schwede/mobius.git
cd mobius
pip install -e .
```

## Quick tutorial

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np
from mobius import PolymerSampler, SequenceGA
from mobius import Map4Fingerprint
from mobius import GPModel, ExpectedImprovement, TanimotoSimilarityKernel
from mobius import LinearPeptideEmulator
from mobius import homolog_scanning, alanine_scanning
from mobius import convert_FASTA_to_HELM


# Simple linear peptide emulator/oracle for MHC class I A*0201. 
# The Position Specific Scoring Matrices (PSSM) can be downloaded from the 
# IEDB database (http://tools.iedb.org/mhci/download/). WARNING: This is 
# for benchmarking purpose only. This step should be an 
actual lab experiment.
pssm_files = ['data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-8.txt',
              'data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-9.txt',
              'data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-10.txt',
              'data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-11.txt']
lpe = LinearPeptideEmulator(pssm_files)

# Now we define a peptide sequence we want to optimize
lead_peptide = convert_FASTA_to_HELM('HMTEVVRRC')[0]

# Then we generate the first seed library of 96 peptides 
# using a combination of both alanine scanning and homolog 
# scanning sequence-based strategies
seed_library = [lead_peptide]

for seq in alanine_scanning(lead_peptide):
    seed_library.append(seq)
    
for seq in homolog_scanning(lead_peptide):
    seed_library.append(seq)

    if len(seed_library) >= 96:
        print('Reach max. number of peptides allowed.')
        break

# The seed library is then virtually tested (Make/Test)
# using the linear peptide emulator we defined earlier.
# WARNING: This is for benchmarking purpose only. This 
# step is supposed to be an actual lab experiment.
pic50_seed_library = lpe.predict(seed_library)

# Once we got results from our first lab experiment
# we can now start the Bayesian Optimization (BO)
# First, we define the molecular fingerprint we want to
# use as well as the surrogate model (Gaussian Process)
# and the acquisition function (Expected Improvement)
map4 = Map4Fingerprint(input_type='helm_rdkit', dimensions=4096, radius=1)
gpmodel = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
ei = ExpectedImprovement(gpmodel, maximize=False)

# ... and also we define the search protocol
# that will be used to search/sample peptide sequences
# optimizing the acquisition function
search_protocol = {
    'SequenceGA': {
        'function': SequenceGA,
        'parameters': {
            'n_process': -1,
            'n_gen': 1000,
            'n_children': 500,
            'temperature': 0.01,
            'elitism': True,
            'total_attempts': 50,
            'cx_points': 2,
            'pm': 0.1,
            'minimum_mutations': 1,
            'maximum_mutations': 5
        }
    }
}

ps = PolymerSampler(ei, search_protocol)

# Now it is time to run the optimization!!
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

The installation instructions, documentation and tutorials can be found on [readthedocs.org](https://mobius.readthedocs.io/en/master/).

## Citation

* [J. Eberhardt, M. Lill, T. Schwede. (2023). Combining Bayesian optimization with sequence- or structure-based strategies for optimization of peptide-binding protein.](https://doi.org/10.26434/chemrxiv-2023-b7l81) (Mobius version used: v0.2; to see all the new features see branch [v0.3](https://git.scicore.unibas.ch/schwede/mobius/-/tree/v0.3?ref_type=heads))
