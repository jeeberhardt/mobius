#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Bayesian optimization
#

import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import gpytorch
import matplotlib.pyplot as plt
from map4 import MAP4Calculator

from mobius.ga import GA
from mobius.baye import DMTSimulation
from mobius.descriptors import Map4Fingerprint, SequenceDescriptors
from mobius.kernels import TanimotoSimilarityKernel
from mobius import expected_improvement, probability_of_improvement, greedy
from mobius import HELMGeneticOperators, compute_probability_matrix
from mobius import MHCIPeptideScorer
from mobius import energy_to_affinity_binding, affinity_binding_to_energy


pssm_files = ['./IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-8.txt',
              './IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-9.txt',
              './IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-10.txt',
              './IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-11.txt']

mhci = pd.read_csv('./binding_data_2013/bdata.20130222.mhci.csv')

# We removed those binding affinity values
# A lot of peptides were set with those values. Looks like some default values assigned...
dirty_values = [1, 2, 3, 5000, 10000, 20000, 43424, 50000, 69444.44444, 78125]

# Split dataset in training and testing sets
mhci = mhci[(mhci['mhc_allele'] == 'HLA-A*02:01') &
            (8 <= mhci['length']) & (mhci['length'] <= 11) &
            (~mhci['affinity_binding'].isin(dirty_values))]

# Genetic operators
with open('HELMCoreLibrary.json') as f:
    helm_core_library = json.load(f)

monomer_names = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
smiles = [x['smiles'] for x in helm_core_library if x['symbol'] in monomer_names and x['monomerType'] == 'Backbone']
probability_matrix = compute_probability_matrix(smiles)

energy_cutoff = -4.11 # 1 mM
#energy_cutoff = -4.944 # 250 uM
#energy_cutoff = -8.235 # 1 uM

mps = MHCIPeptideScorer(pssm_files, mhci, energy_cutoff=energy_cutoff)
map4 = Map4Fingerprint(input_type='helm')
seq_descriptors = SequenceDescriptors(pd.read_csv('pca.csv'), input_type='helm')
helmgo = HELMGeneticOperators(monomer_names, probability_matrix)

with open('save.pkl', 'wb') as f:
    pickle.dump(mps, f)

with open('save.pkl', 'wb') as f:
    pickle.dump(helmgo, f)

with open('save.pkl', 'wb') as f:
    pickle.dump(MAP4Calculator, f)

#with open('save.pkl', 'wb') as f:
#    pickle.dump(map4, f)

n_peptides = 100
peptide_lengths = [8, 9, 10, 11]
energy_bounds = [-8.235, -4.944] # about between 1 uM and 250 uM
#energy_bounds = [-4.944, -4.531] # about between 250 uM and 500 uM
#energy_bounds = [-4.531, -4.118] # about between 500 uM and 1 mM
#energy_bounds = [-8.649, -8.235] # about between 500 nM and 1uM

random_peptides, random_peptide_energies = mps.generate_random_peptides(n_peptides, peptide_lengths, energy_bounds)

clusters = defaultdict(list)
for c, sequence in enumerate(random_peptides):
    clusters[sequence.count('.')].append(c)
print('Distribution:', ['%d: %d' % (k, len(clusters[k])) for k in sorted(clusters.keys())])
print('')

parameters = {'n_candidates': 96, 'n_gen': 1, 'GA': GA, 'helmgo': helmgo, 'oracle': mps, 'acq_function': expected_improvement, 'kernel': TanimotoSimilarityKernel(), 'seq_transformer': map4, 
              'sequence_n_gen': 1000, 'sequence_n_children': 100, 'sequence_temperature': 0.025, 'sequence_elitism': True, 'sequence_total_attempts': 20, 
              'sequence_minimum_mutations': 1, 'sequence_maximum_mutations': 3, 'sequence_n_process': 4,
              'scaffold_n_gen': 1, 'scaffold_n_children': 100, 'scaffold_temperature': 0.05, 'scaffold_elitism': True, 'scaffold_total_attempts': 20, 
              'scaffold_only_terminus': True, 'scaffold_minimum_size': 8, 'scaffold_maximum_size': 11}

dmt = DMTSimulation(5, 10)
df = dmt.run(random_peptides, random_peptide_energies, **parameters)
df.to_csv('100-8-9-10-11-mers_SeqGA_ei_tanimoto_map4.csv', index=False)
