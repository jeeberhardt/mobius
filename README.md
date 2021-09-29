# Mobius

## Requirements
* Python (>= 3.5)
* Numpy
* Pandas

## Installation

```bash
git clone https://git.scicore.unibas.ch/schwede/mobius.git
cd mobius
python setupy.py install
```

## Quick tutorial

```python
from mobius import VirtualTarget, ForceField

ff = ForceField()
vt = VirtualTarget(ff)

# Generate a random pharmacophore for a 6-mer peptide
vt.generate_random_pharmacophore(6)
print(vt)

# Generate some random sequence related to the pharmacophore
# They can be used as a seed to benchmark search algorithms
parents = vt.generate_random_peptides_from_pharmacophore(10, sigmas=[0, 0.1, 0.1])
# ... and score them based on the solvent exposure, hydrophilicity and volume
parents_score = vt.score_peptides(parents)

print(parents)
print(parents_score)

# Save the pharmacophore, can be loaded using 'load_pharmacophore' function
vt.export_pharmacophore('pharmacophore.csv')
```
