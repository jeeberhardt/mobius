#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MHC-I
#

import json
import time

import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mobius import SequenceGA, ScaffoldGA, GA
from mobius import Map4Fingerprint, SequenceDescriptors, TanimotoSimilarityKernel, DMTSimulation
from mobius import expected_improvement, probability_of_improvement, greedy
from mobius import HELMGeneticOperators, compute_probability_matrix
from mobius import MHCIPeptideScorer


# MHCI dataset
energy_cutoff = -4.11 # 1 mM

pssm_files = ['./IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-8.txt',
              './IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-9.txt',
              './IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-10.txt',
              './IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-11.txt']

mhci = pd.read_csv('./binding_data_2013/bdata.20130222.mhci.csv')

# We removed those binding affinity values
# A lot of peptides were set with those values. Looks like some default values assigned...
dirty_values = [1, 2, 3, 5000, 10000, 20000, 43424, 50000, 69444.44444, 78125]
mhci = mhci[(mhci['mhc_allele'] == 'HLA-A*02:01') &
            (8 <= mhci['length']) & (mhci['length'] <= 11) &
            (~mhci['affinity_binding'].isin(dirty_values))]

# Genetic operators
with open('HELMCoreLibrary.json') as f:
    helm_core_library = json.load(f)

monomer_names = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
smiles = [x['smiles'] for x in helm_core_library if x['symbol'] in monomer_names and x['monomerType'] == 'Backbone']
probability_matrix = compute_probability_matrix(smiles)


mps = MHCIPeptideScorer(pssm_files, mhci, energy_cutoff=energy_cutoff)
map4 = Map4Fingerprint(input_type='helm')
helmgo = HELMGeneticOperators(monomer_names, probability_matrix)

parameters = {'n_candidates': 96, 'GA': SequenceGA, 'helmgo': helmgo, 'oracle': mps,
              'acq_function': expected_improvement, 'kernel': TanimotoSimilarityKernel(), 'seq_transformer': map4, 
              'n_gen': 100000, 'n_children': 200, 'temperature': 0.025, 'elitism': True, 'total_attempts': 20, 'minimum_mutations': 1, 'maximum_mutations': 3}

dmt = DMTSimulation(5, 10)

for i in range(10):
    dataset = pd.read_csv('dataset_peptides_100_9_%d.csv' % i)

    start = time.time()
    df = dmt.run(dataset['sequence'].values, dataset['energy'].values, **parameters)
    print('Time elapsed: %s seconds' % (time.time() - start)) 

    run_name = '10-9-mers_SeqGA_ei_tanimoto_map4_temp-0.025_pop-200_%d' % i
    df.to_csv('data_%s.csv' % run_name, index=False)
