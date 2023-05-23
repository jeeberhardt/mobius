#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic Operators
#


import numpy as np

from .utils import build_helm_string, parse_helm, get_scaffold_from_helm_string


class GeneticOperators:
    """
    A class for applying genetic operators on sequences in HELM format.

    """
    
    def __init__(self, seed=None):
        """
        Initialize the GeneticOperators class.

        Parameters
        ----------
        seed: int, default : None
            Seed value for the random number generator.

        """        
        self._random_seed = seed
        self._rng = np.random.default_rng(self._random_seed)
        self._connections_dtype = [('SourcePolymerID', 'U20'), ('TargetPolymerID', 'U20'),
                                   ('SourceMonomerPosition', 'i4'), ('SourceAttachment', 'U2'),
                                   ('TargetMonomerPosition', 'i4'), ('TargetAttachment', 'U2')]

    def mutate(self, input_sequence, scaffold_designs, n=10, 
               minimum_mutations=1, maximum_mutations=None, keep_connections=True):
        """
        Apply the mutation genetic operator on the input sequence.

        Parameters
        ----------
        input_sequence : str
            Input sequence in HELM format.
        scaffold_designs : dictionary
            Dictionary with scaffold sequences and defined set of monomers to 
            use for each position.
        n : int, default : 10
            Total number of sequences to generate.
        minimum_mutations : int, default : 1
            The minimum number of monomers to mutate in each sequence.
        maximum_mutations : int, default : None
            The maximum number of monomers to mutate in each sequence.
        keep_connections : bool, default : True
            If True, monomers involved in a connection won't be mutated. If False, 
            monomers involved in a connection can be mutated. As consequence, the
            connections will be deleted.

        Returns
        -------
        List of str
            Mutated sequences in HELM format.

        """
        mutant_sequences = []

        scaffold = get_scaffold_from_helm_string(input_sequence)
        assert scaffold in scaffold_designs, 'Scaffold %s not found in the scaffold designs.' % scaffold

        # Get the scaffold design for the current scaffold
        scaffold_design = scaffold_designs[scaffold]
        
        polymers, connections, _, _ = parse_helm(input_sequence)

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
                    # -1, because positions are 1-based in HELM
                    connection_resids = np.asarray(connection_resids) - 1
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
                    # +1 , because positions are 1-based in HELM
                    chosen_monomer = self._rng.choice(scaffold_design[pid][mutation_position + 1])
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
                mutant_sequence = build_helm_string({p: s[0] for p, s in mutant_polymers.items()}, connections[connections_to_keep])
                mutant_sequences.append(mutant_sequence)
            else:
                mutant_sequences.append(input_sequence)
        
        return mutant_sequences

    def crossover(self, input_sequence1, input_sequence2, cx_points=2):
        """
        Apply crossover genetic operator on the input sequence.

        Parameters
        ----------
        input_sequence1 : str
            The first parent sequence in HELM format.
        input_sequence2 : str
            The second parent sequence in HELM format.
        cx_points : int, default : 2
            The number of crossover points.

        Returns
        -------
        List of str
            Two mutated sequences (children) in HELM format.

        Raises
        ------
        ValueError
            If polymers in the two input HELM strings have different polymer ids or different lengths, 
            or if the scaffold (connections) in the two input HELM strings are different.

        """
        mutant1_polymers = {}
        mutant2_polymers = {}
        mutant_sequences = []

        polymers1, connections1, _, _ = parse_helm(input_sequence1)
        polymers2, connections2, _, _ = parse_helm(input_sequence2)

        if polymers1.keys() != polymers2.keys():
            raise ValueError('Polymers do not contain the same polymer ids.')

        if connections1 != connections2:
            raise ValueError('Polymers must have the same scaffold (connections are different).')

        for pid in polymers1.keys():
            if len(polymers1[pid]) != len(polymers2[pid]):
                raise ValueError('Polymer sequences with ID %s have different lengths.' % pid)

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

        mutant_sequences.extend([build_helm_string(mutant1_polymers, connections1),
                                 build_helm_string(mutant2_polymers, connections2)])

        return mutant_sequences
