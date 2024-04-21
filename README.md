# Mobius

A python package for optimizing peptide sequences using Bayesian optimization (BO).

## Features

- Single or multi-objectives optimization, with constraints
- Supports linear and non-linear peptide sequences (macrocyclic, lasso, branched, etc, ...)
- Supports any non-standard amino-acid residues and modifications
- Use design protocols to customize the sequence optimization
- Integrate protein or chemical Language Models, and fine-tune them on existing data (classical or LoRA).
- Easily extensible to add your own molecular representations or other ML models
- Can be seamlessly integrated with other tools (docking, pyRosetta, AlphaFold, etc..)

## Documentation

The installation instructions, documentation and tutorials can be found at [mobius.readthedocs.io](https://mobius.readthedocs.io/en/latest/about.html).

## Installation

First, you need to download the python package:
```bash
git clone https://git.scicore.unibas.ch/schwede/mobius.git
```

Now you can set up the environment using `mamba`. For more instructions on how to install `mamba`, see: https://github.com/conda-forge/miniforge#miniforge3 
```bash
cd mobius
mamba env create -f environment.yaml -n mobius
mamba activate mobius
```

Once you created the environment, you can install the package using the following command:
```bash
pip install -e .
```

## Quick tutorial

```python
import numpy as np
from mobius import Map4Fingerprint
from mobius import GPModel, ExpectedImprovement, TanimotoSimilarityKernel
from mobius import LinearPeptideEmulator
from mobius import homolog_scanning
from mobius import convert_FASTA_to_HELM
from mobius import Planner, SequenceGA
```

Simple linear peptide emulator/oracle for MHC class I A*0201. The Position Specific Scoring Matrices
(PSSM) can be downloaded from the [IEDB](http://tools.iedb.org/mhci/download/) database (see `Scoring 
matrices of SMM and SMMPMBEC` section). WARNING: This is for benchmarking purpose only. This step should be an 
actual lab experiment.
```python
lpe = LinearPeptideEmulator('IEDB_MHC/smmpmbec_matrix/HLA-A-02:01-9.txt',)
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
pic50_seed_library = lpe_one.score(seed_library)
```

Once we have the results from our first lab experiment we can now start the Bayesian Optimization (BO). First, 
we define the molecular fingerprint we want to use as well as the surrogate model (Gaussian Process) 
and the acquisition function (Expected Improvement).
```python
map4 = Map4Fingerprint(input_type='helm', dimensions=4096, radius=1)
gpmodel = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
acq_fun = ExpectedImprovement(gpmodel, maximize=[False, False])
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
  polymers:
    - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0:
        PEPTIDE1:
          1: [AROMATIC, NEG_CHARGED]
          4: POLAR
          9: [A, V, I, L, M, T]

```

Once the acquisition function is defined and the parameters set in the YAML configuration file, we can initiate 
the single-objective problem we are optimising for and the planner method.
```python
optimizer = SequenceGA(algorithm='GA', period=15, design_protocol_filename='design_protocol.yaml')
planner = Planner(acq_fun, optimizer)
```

Now it is time to run the optimization!!
```python
peptides = seed_library.copy()
pic50_scores = pic50_seed_library.copy()

# Here we are going to do 3 DMT cycles
for i in range(3):
    # Run optimization, recommand 96 new peptides based on existing data
    suggested_peptides, _ = ps.recommand(peptides, pic50_scores, batch_size=96)

    # Here you can add whatever methods you want to further filter out peptides
    
    # Get the pIC50 (Make/Test) of all the suggested peptides using the MHC emulator
    # WARNING: This is for benchmarking purpose only. This 
    # step is supposed to be an actual lab experiment.
    pic50_suggested_peptides = lpe_one.score(suggested_peptides)
    
    # Add all the new data
    peptides = np.concatenate([peptides, suggested_peptides])
    pic50_scores = np.concatenate((pic50_scores, pic50_suggested_peptides), axis=0)
```

## Citation

If mobius is useful for your work please cite the following [paper](https://doi.org/10.26434/chemrxiv-2023-b7l81-v2):

```bibtex
@article{eberhardt2024combining,
  title={Combining Bayesian optimization with sequence-or structure-based strategies for optimization of protein-peptide binding},
  author={Eberhardt, Jerome and Lees, Aidan and Lill, Markus and Schwede, Torsten},
  year={2024}
}
```
