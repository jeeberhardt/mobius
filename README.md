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
vt.generate_random_target_sequence(6)
print(vt)

# Generate some random sequence related to the pharmacophore
# They can be used as a seed to benchmark search algorithms
parents = vt.generate_random_peptides_from_target_sequence(10, maximum_mutations=3)
# ... and score them based on the solvent exposure, hydrophilicity, volume and net charge
parents_score = vt.score_peptides(parents)

print(parents)
print(parents_score)

# Save the virtual target
# Can be loaded using 'vt = VirtualTarget.load_virtual_target('virtual_target.pkl')' function
vt.export_virtual_target('virtual_target.pkl')
```
