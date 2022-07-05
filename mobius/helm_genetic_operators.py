#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# HELM
#

import json
import os
import random

import numpy as np
import torch

from . import utils
from .descriptors import Map4Fingerprint
from .kernels import TanimotoSimilarityKernel
from .helm import build_helm_string, parse_helm


def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur.
    
    Source: https://stackoverflow.com/questions/3589214/generate-random-numbers-summing-to-a-predefined-value

    """

    dividers = sorted(random.sample(range(1, total), n - 1))
    return np.array([a - b for a, b in zip(dividers + [total], [0] + dividers)])


def constrained_sum_sample_nonneg(n, total):
    """Return a randomly chosen list of n nonnegative integers summing to total.
    Each such list is equally likely to occur.

    Source: https://stackoverflow.com/questions/3589214/generate-random-numbers-summing-to-a-predefined-value

    """

    return np.array([x - 1 for x in constrained_sum_sample_pos(n, total + n)])


def compute_probability_matrix(smiles):
    probability_matrix = []

    map4calc = Map4Fingerprint(input_type='smiles')
    fps = torch.from_numpy(map4calc.transform(smiles)).float()
    
    t = TanimotoSimilarityKernel()
    similarity_matrix = t.forward(fps, fps).numpy()

    for aa in similarity_matrix:
        tmp = aa.copy()
        tmp[tmp == 1.0] = 0
        probability_matrix.append(tmp / np.sum(tmp))

    probability_matrix = np.array(probability_matrix)
    
    return probability_matrix


class HELMGeneticOperators:
    
    def __init__(self, monomer_symbols=None, HELMCoreLibrary=None, seed=None):
        if monomer_symbols is None:
            # If we do not provide any monomer symbols, the 20 canonical amino acids will be used
            self._monomer_symbols = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
                                     "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        else:
            self._monomer_symbols = monomer_symbols

        # Load HELMCore library
        if HELMCoreLibrary is None:
            d = utils.path_module("mobius")
            helmcorelibrary_filename = os.path.join(d, "data/HELMCoreLibrary.json")
        else:
            helmcorelibrary_filename = HELMCoreLibrary

        with open(helmcorelibrary_filename) as f:
            helm_core_library = json.load(f)

        # Get SMILES from all monomers, only peptide
        smiles = [x['smiles'] for x in helm_core_library if x['symbol'] in self._monomer_symbols and x['polymerType'] == 'PEPTIDE']

        # Compute probability matrix
        self._probability_matrix = compute_probability_matrix(smiles)
        
        self._random_seed = seed
        self._rng = np.random.default_rng(self._random_seed)
        self._connections_dtype = [('SourcePolymerID', 'U20'), ('TargetPolymerID', 'U20'),
                                   ('SourceMonomerPosition', 'i4'), ('SourceAttachment', 'U2'),
                                   ('TargetMonomerPosition', 'i4'), ('TargetAttachment', 'U2')]
        
    def insert(self, helm_string, n=10, only_terminus=False, maximum_size=None):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)

        if connections:
            raise NotImplementedError('Does not handle peptides with connections.')
        
        # Generate mutants...
        for i in range(n):
            mutant_polymers = {}
            total_mutations = 0
        
            for pid, sequence in polymers.items():
                mutated_sequence = list(sequence)

                # Choose positions where to insert new mutations
                if only_terminus:
                    mutation_positions = np.array([0, len(sequence)])
                else:
                    mutation_positions = np.array(range(len(sequence)))

                # Define the max number of mutations that will be inserted
                if maximum_size is not None:
                    # In the case where mutation_positions is smaller than diff maximum_size and sequence
                    maximum_mutations = maximum_size - len(sequence)
                else:
                    maximum_mutations = mutation_positions.shape[0]

                # Choose random number of mutations per position
                if maximum_mutations > 0:
                    # There will always be at least one mutation
                    try:
                        n_mutations = np.random.randint(1, high=maximum_mutations + 1)
                    except ValueError:
                        n_mutations = 1

                    mutations_per_position = constrained_sum_sample_nonneg(mutation_positions.shape[0], n_mutations)
                else:
                    # We move to the next polymer since the sequence length exceed already the maximum length allowed
                    mutant_polymers[pid] = (sequence, np.array([]))
                    continue

                # Keep positions where there are mutations
                mutation_positions = mutation_positions[mutations_per_position > 0]
                mutations_per_position = mutations_per_position[mutations_per_position > 0]

                for mutation_position, n_mutations in zip(mutation_positions[::-1], mutations_per_position[::-1]):
                    for _ in range(n_mutations):
                        chosen_monomer = self._rng.choice(self._monomer_symbols)
                        mutated_sequence.insert(mutation_position, chosen_monomer)

                # Stored mutated sequence, number of mutations per position and where they were inserted
                mutant_polymers[pid] = (''.join(mutated_sequence), (mutation_positions, mutations_per_position))
                total_mutations += np.sum(mutations_per_position)

            if total_mutations > 0:
                # Reconstruct the HELM string
                mutant_helm_string = build_helm_string({p: s[0] for p, s in mutant_polymers.items()})
                mutant_helm_strings.append(mutant_helm_string)
            else:
                mutant_helm_strings.append(helm_string)
        
        return mutant_helm_strings

    def delete(self, helm_string, n=10, only_terminus=False, minimum_size=None):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)

        if connections:
            raise NotImplementedError('Does not handle peptides with connections.')
        
        # Generate mutants...
        for i in range(n):
            mutant_polymers = {}
            total_mutations = 0
        
            for pid, sequence in polymers.items():
                mutated_sequence = list(sequence)

                # Choose positions where to insert new mutations
                if only_terminus:
                    mutation_positions = np.array([0, len(sequence) - 1])
                else:
                    mutation_positions = np.array(range(len(sequence)))

                # Define the max number of mutations that will be inserted
                if minimum_size is not None:
                    # In the case where mutation_positions is smaller than diff maximum_size and sequence
                    maximum_mutations = len(sequence) - minimum_size
                else:
                    maximum_mutations = mutation_positions.shape[0]

                # Choose random number of mutations per position
                if maximum_mutations > 0:
                    # There will always be at least one mutation
                    try:
                        n_mutations = np.random.randint(1, high=maximum_mutations + 1)
                    except ValueError:
                        n_mutations = 1

                    mutations_per_position = constrained_sum_sample_nonneg(mutation_positions.shape[0], n_mutations)

                    # FOR NOW, WE LIMIT TO ONE DELETION PER POSITION BECAUSE OF THE CONNECTIONS. Need to figure this out...
                    mutations_per_position[mutations_per_position > 1] = 1
                else:
                    # We move to the next polymer since the sequence length exceed already the maximum length allowed
                    mutant_polymers[pid] = (sequence, np.array([]))
                    continue

                # Keep positions where there are mutations
                mutation_positions = mutation_positions[mutations_per_position > 0]
                mutations_per_position = mutations_per_position[mutations_per_position > 0]

                for mutation_position, n_mutations in zip(mutation_positions[::-1], mutations_per_position[::-1]):
                    for _ in range(n_mutations):
                        mutated_sequence.pop(mutation_position)

                mutant_polymers[pid] = (''.join(mutated_sequence), (mutation_positions, mutations_per_position))
                total_mutations += np.sum(mutations_per_position)

            if total_mutations > 0:
                # Reconstruct the HELM string
                mutant_helm_string = build_helm_string({p: s[0] for p, s in mutant_polymers.items()})
                mutant_helm_strings.append(mutant_helm_string)
            else:
                mutant_helm_strings.append(helm_string)
        
        return mutant_helm_strings

    def mutate(self, helm_string, n=10, minimum_mutations=1, maximum_mutations=None):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)

        if connections:
            raise NotImplementedError('Does not handle peptides with connections.')
        
        # Generate mutants...
        for i in range(n):
            mutant_polymers = {}
            n_mutations = 0
            
            for pid, sequence in polymers.items():
                mutated_sequence = list(sequence)
                possible_positions = list(range(len(sequence)))
                
                # Choose a random number of mutations between min and max
                if minimum_mutations == maximum_mutations:
                    number_mutations = maximum_mutations
                elif maximum_mutations is None:
                    number_mutations = self._rng.integers(low=minimum_mutations, high=len(sequence))
                else:
                    number_mutations = self._rng.integers(low=minimum_mutations, high=maximum_mutations)
                
                # Choose positions to mutate
                mutation_positions = self._rng.choice(possible_positions, size=number_mutations, replace=False)
                
                # Do mutations
                for mutation_position in mutation_positions:                    
                    monomer_symbol = mutated_sequence[mutation_position]
                    
                    # Force mutation!
                    while monomer_symbol == mutated_sequence[mutation_position]:
                        # Instead of mutating to random monomer, the selection will be
                        # based on the probability matrix
                        if self._probability_matrix is not None:
                            index_symbol = self._monomer_symbols.index(monomer_symbol)
                            p = self._probability_matrix[index_symbol]
                        else:
                            p = None
                        
                        chosen_monomer = self._rng.choice(self._monomer_symbols, p=p)
                        mutated_sequence[mutation_position] = chosen_monomer
                
                mutant_polymers[pid] = (mutated_sequence, mutation_positions)
                n_mutations += len(mutation_positions)
            
            if n_mutations > 0:
                # Reconstruct the HELM string
                mutant_helm_string = build_helm_string({p: s[0] for p, s in mutant_polymers.items()})
                mutant_helm_strings.append(mutant_helm_string)
            else:
                mutant_helm_strings.append(helm_string)
        
        return mutant_helm_strings

    def crossover(self, helm_string1, helm_string2, cx_points=2):
        mutant1_polymers = {}
        mutant2_polymers = {}
        mutant_helm_strings = []
            
        polymers1, connections1, _, _ = parse_helm(helm_string1)
        polymers2, connections2, _, _ = parse_helm(helm_string2)
        
        if connections1 or connections2:
            raise NotImplementedError('Does not handle peptides with connections.')

        if polymers1.keys() != polymers2.keys():
            raise RuntimeError('Peptides does not contain the same polymer ids.')
        
        for pid in polymers1.keys():
            if len(polymers1[pid]) != len(polymers2[pid]):
                raise RuntimeError('Peptide sequence lengths are different.')
            
            # Copy parents since we are going to modify the children
            ind1 = list(polymers1[pid])
            ind2 = list(polymers2[pid])
            
            # Choose positions to crossover
            possible_positions = list(range(len(polymers1[pid])))
            cx_positions = self._rng.choice(possible_positions, size=cx_points, replace=False)
            cx_positions = np.sort(cx_positions)
            
            for cx_position in cx_positions:
                ind1[cx_position:], ind2[cx_position:] = ind2[cx_position:], ind1[cx_position:]
                
            mutant1_polymers[pid] = ''.join(ind1)
            mutant2_polymers[pid] = ''.join(ind2)
            
        mutant_helm_strings.extend([build_helm_string(mutant1_polymers), build_helm_string(mutant2_polymers)])
        
        return mutant_helm_strings
