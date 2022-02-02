#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# HELM
#

import numpy as np

from helm import build_helm_string, parse_helm


class HELMGeneticOperators:
    
    def __init__(self, monomer_library, probability_matrix=None, seed=None):
        self._monomer_library = monomer_library
        self._monomer_symbols = [m['symbol'] for m in self._monomer_library]
        self._probability_matrix = probability_matrix
        
        self._random_seed = seed
        self._rng = np.random.default_rng(self._random_seed)
        self._connections_dtype = [('SourcePolymerID', 'U20'), ('TargetPolymerID', 'U20'),
                                   ('SourceMonomerPosition', 'i4'), ('SourceAttachment', 'U2'),
                                   ('TargetMonomerPosition', 'i4'), ('TargetAttachment', 'U2')]
        
    def insert(self, helm_string, n=10, probability=0.05, only_terminus=False, maximum_size=None):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)
        
        # Generate mutants...
        for i in range(n):
            mutant_polymers = {}
            n_mutations = 0
        
            for pid, sequence in polymers.items():
                mutated_sequence = list(sequence)

                if only_terminus:
                    possible_positions = np.array([0, len(sequence)])
                else:
                    possible_positions = np.array(range(len(sequence)))

                # Choose positions to insert
                mutation_probabilities = self._rng.uniform(size=len(possible_positions))
                mutation_positions = possible_positions[mutation_probabilities <= probability]

                # We move to the next polymer if there is no monomer to insert...
                if len(mutation_positions) == 0:
                    mutant_polymers[pid] = (sequence, np.array([]))
                    continue

                if maximum_size is not None:
                    # In the case where mutation_positions is smaller than diff maximum_size and sequence
                    mutations_to_select = np.min([maximum_size - len(sequence), len(mutation_positions)])

                    if mutations_to_select > 0:
                        mutation_positions = self._rng.choice(mutation_positions, size=mutations_to_select, replace=False)
                    else:
                        # We move to the next polymer since the sequence length exceed already the maximum length allowed
                        mutant_polymers[pid] = (sequence, np.array([]))
                        continue

                # Since we are going to insert from the end, the array must be sorted
                mutation_positions = np.sort(mutation_positions)

                for mutation_position in mutation_positions[::-1]:
                    chosen_monomer = self._monomer_library[self._rng.choice(len(self._monomer_library))]['symbol']
                    mutated_sequence.insert(mutation_position, chosen_monomer)

                mutant_polymers[pid] = (''.join(mutated_sequence), mutation_positions)
                n_mutations += len(mutation_positions)

            if n_mutations > 0:
                data = []

                # Shift attachment positions
                for i, connection in enumerate(connections):
                    source_mutations = mutant_polymers[connection['SourcePolymerID']][1]
                    target_mutations = mutant_polymers[connection['TargetPolymerID']][1]

                    source_position = connection['SourceMonomerPosition'] + np.sum([source_mutations < connection['SourceMonomerPosition']])
                    target_position = connection['TargetMonomerPosition'] + np.sum([target_mutations < connection['TargetMonomerPosition']])

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
        
    def remove(self, helm_string, n=10, probability=0.05, only_terminus=False, minimun_size=None, keep_connections=True):
        mutant_helm_strings = []
        
        polymers, connections, _, _ = parse_helm(helm_string)
        
        # Generate mutants...
        for i in range(n):
            mutant_polymers = {}
            n_mutations = 0
        
            for pid, sequence in polymers.items():
                mutated_sequence = list(sequence)

                if only_terminus:
                    possible_positions = [0, len(sequence) - 1]
                else:
                    possible_positions = range(len(sequence))

                # Residues involved in a connection within and between polymers won't be removed
                if keep_connections and pid in polymers.keys():
                    connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
                    connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
                    # Because positions are 1-based in HELM
                    connection_resids = np.array(connection_resids) - 1
                    possible_positions = list(set(possible_positions).difference(connection_resids))

                # Choose positions to remove
                possible_positions = np.array(possible_positions)
                mutation_probabilities = self._rng.uniform(size=len(possible_positions))
                mutation_positions = possible_positions[mutation_probabilities <= probability]

                # We move to the next polymer if there is no monomer to remove...
                if len(mutation_positions) == 0:
                    mutant_polymers[pid] = (sequence, np.array([]))
                    continue

                if minimun_size is not None:
                    # In the case where mutation_positions is smaller than diff maximum_size and sequence
                    mutations_to_select = np.min([len(sequence) - minimun_size, len(mutation_positions)])

                    if mutations_to_select > 0:
                        mutation_positions = self._rng.choice(mutation_positions, size=mutations_to_select, replace=False)
                    else:
                        # We move to the next polymer since the sequence length exceed already the maximum length allowed
                        mutant_polymers[pid] = (sequence, np.array([]))
                        continue

                # Since we are going to remove from the end, the array must be sorted
                mutation_positions = np.sort(mutation_positions)

                for mutation_position in mutation_positions[::-1]:
                    mutated_sequence.pop(mutation_position)

                mutant_polymers[pid] = (''.join(mutated_sequence), mutation_positions)
                n_mutations += len(mutation_positions)

            if n_mutations > 0:
                connections_to_keep = []
                data = []

                # Check if we have to remove connections due to the monomers removed
                for i, connection in enumerate(connections):
                    # The connection positions must not be in the mutation lists
                    # mutant_polymers[connection['XXXXXPolymerID']][1] + 1 because positions are 1-based in HELM
                    if connection['SourceMonomerPosition'] not in mutant_polymers[connection['SourcePolymerID']][1] + 1 and \
                       connection['TargetMonomerPosition'] not in mutant_polymers[connection['TargetPolymerID']][1] + 1:
                        connections_to_keep.append(i)

                # Shift attachment positions (only the connections to keep)
                for i, connection in enumerate(connections[connections_to_keep]):
                    source_mutations = mutant_polymers[connection['SourcePolymerID']][1]
                    target_mutations = mutant_polymers[connection['TargetPolymerID']][1]

                    source_position = connection['SourceMonomerPosition'] - np.sum([source_mutations < connection['SourceMonomerPosition']])
                    target_position = connection['TargetMonomerPosition'] - np.sum([target_mutations < connection['TargetMonomerPosition']])

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
