#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# HELM
#

import random

import numpy as np

from helm import build_helm_string, parse_helm


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


def compute_probability_matrix(smiles, kernel=None):
    probability_matrix = []
    
    if kernel is None:
        kernel = TanimotoSimilarityKernel

    fps = map4_fingerprint(smiles, input_type='smiles', radius=2)
    
    t = kernel()
    similarity_matrix = t.forward(fps, fps).numpy()

    for aa in similarity_matrix:
        tmp = aa.copy()
        tmp[tmp == 1.0] = 0
        probability_matrix.append(tmp / np.sum(tmp))

    probability_matrix = np.array(probability_matrix)
    
    return probability_matrix


class HELMGeneticOperators:
    
    def __init__(self, monomer_symbols, probability_matrix=None, seed=None):
        self._monomer_symbols = monomer_symbols
        self._probability_matrix = probability_matrix
        
        self._random_seed = seed
        self._rng = np.random.default_rng(self._random_seed)
        self._connections_dtype = [('SourcePolymerID', 'U20'), ('TargetPolymerID', 'U20'),
                                   ('SourceMonomerPosition', 'i4'), ('SourceAttachment', 'U2'),
                                   ('TargetMonomerPosition', 'i4'), ('TargetAttachment', 'U2')]
        
    def insert(self, helm_string, n=10, only_terminus=False, maximum_size=None):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)
        
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
                data = []

                # Shift attachment positions
                for i, connection in enumerate(connections):
                    source_mutations = mutant_polymers[connection['SourcePolymerID']][1]
                    target_mutations = mutant_polymers[connection['TargetPolymerID']][1]

                    source_position = connection['SourceMonomerPosition'] \
                                    + np.sum(source_mutations[1][source_mutations[0] < connection['SourceMonomerPosition']]) \

                    target_position = connection['TargetMonomerPosition'] \
                                    + np.sum(target_mutations[1][target_mutations[0] < connection['TargetMonomerPosition']])

                    data.append((connection['SourcePolymerID'], connection['TargetPolymerID'], 
                                 source_position, connection['SourceAttachment'],
                                 target_position, connection['TargetAttachment']))

                new_connections = np.array(data, dtype=self._connections_dtype)

                # Reconstruct the HELM string
                mutant_helm_string = build_helm_string({p: s[0] for p, s in mutant_polymers.items()}, new_connections)

                mutant_helm_strings.append(mutant_helm_string)
            else:
                mutant_helm_strings.append(helm_string)
        
        return mutant_helm_strings

    def delete(self, helm_string, n=10, only_terminus=False, minimum_size=None, keep_connections=True):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)
        
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

                # Residues involved in a connection within and between polymers won't be removed
                if keep_connections and pid in polymers.keys():
                    connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
                    connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
                    # Because positions are 1-based in HELM
                    connection_resids = np.array(connection_resids) - 1
                    mutation_positions = np.array(list(set(mutation_positions).difference(connection_resids)))

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
                connections_to_keep = []
                data = []

                # Check if we have to remove connections due to the monomers removed
                for i, connection in enumerate(connections):
                    # The connection positions must not be in the mutation lists
                    # mutant_polymers[connection['XXXXXPolymerID']][1] + 1 because positions are 1-based in HELM
                    if connection['SourceMonomerPosition'] not in mutant_polymers[connection['SourcePolymerID']][1][0] + 1 and \
                       connection['TargetMonomerPosition'] not in mutant_polymers[connection['TargetPolymerID']][1][0] + 1:
                        connections_to_keep.append(i)

                # Shift attachment positions (only the connections to keep)
                for i, connection in enumerate(connections[connections_to_keep]):
                    source_mutations = mutant_polymers[connection['SourcePolymerID']][1]
                    target_mutations = mutant_polymers[connection['TargetPolymerID']][1]

                    source_position = connection['SourceMonomerPosition'] \
                                    - np.sum(source_mutations[1][source_mutations[0] < connection['SourceMonomerPosition']])
                    target_position = connection['TargetMonomerPosition'] \
                                    - np.sum(target_mutations[1][target_mutations[0] < connection['TargetMonomerPosition']])

                    data.append((connection['SourcePolymerID'], connection['TargetPolymerID'], 
                                 source_position, connection['SourceAttachment'],
                                 target_position, connection['TargetAttachment']))

                new_connections = np.array(data, dtype=self._connections_dtype)

                # Reconstruct the HELM string
                mutant_helm_string = build_helm_string({p: s[0] for p, s in mutant_polymers.items()}, new_connections)

                mutant_helm_strings.append(mutant_helm_string)
            else:
                mutant_helm_strings.append(helm_string)
        
        return mutant_helm_strings

    def mutate(self, helm_string, n=10, minimum_mutations=1, maximum_mutations=None, keep_connections=True):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)
        
        # Generate mutants...
        for i in range(n):
            mutant_polymers = {}
            n_mutations = 0
            
            for pid, sequence in polymers.items():
                mutated_sequence = list(sequence)
                
                # Residues involved in a connection within and between peptides won't be mutated
                if keep_connections and pid in polymers.keys():
                    connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
                    connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
                    # Because positions are 1-based in HELM
                    connection_resids = np.array(connection_resids) - 1
                    
                    possible_positions = list(set(range(len(sequence))).difference(connection_resids))
                else:
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
                if not keep_connections:
                    connections_to_keep = []

                    # Check if we have to remove connections due to the mutations
                    for i, connection in enumerate(connections):
                        # The connection positions must not be in the mutation lists
                        # mutant_polymers[connection['XXXXXPolymerID']][1] + 1 because positions are 1-based in HELM
                        if connection['SourceMonomerPosition'] not in mutant_polymers[connection['SourcePolymerID']][1] + 1 and \
                           connection['TargetMonomerPosition'] not in mutant_polymers[connection['TargetPolymerID']][1] + 1:
                            connections_to_keep.append(i)
                else:
                    connections_to_keep = list(range(connections.shape[0]))

                # Reconstruct the HELM string
                mutant_helm_string = build_helm_string({p: s[0] for p, s in mutant_polymers.items()}, connections[connections_to_keep])
                mutant_helm_strings.append(mutant_helm_string)
            else:
                mutant_helm_strings.append(helm_string)
        
        return mutant_helm_strings
