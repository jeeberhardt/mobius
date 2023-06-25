#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic Operators
#


import numpy as np

from ..utils import build_helm_string, parse_helm, get_scaffold_from_helm_string


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

    def mutate(self, polymer, scaffold_designs, n=10, 
               minimum_mutations=1, maximum_mutations=None, keep_connections=True):
        """
        Apply the mutation genetic operator on the input polymer.

        Parameters
        ----------
        polymer : str
            Polymer in HELM format.
        scaffold_designs : dictionary
            Dictionary with scaffold polymers and defined set of monomers to 
            use for each position.
        n : int, default : 10
            Total number of polymers to generate.
        minimum_mutations : int, default : 1
            The minimum number of monomers to mutate in each polymer.
        maximum_mutations : int, default : None
            The maximum number of monomers to mutate in each polymer.
        keep_connections : bool, default : True
            If True, monomers involved in a connection won't be mutated. If False, 
            monomers involved in a connection can be mutated. As consequence, the
            connections will be deleted.

        Returns
        -------
        List of str
            Mutated polymers in HELM format.

        """
        mutant_polymers = []

        scaffold = get_scaffold_from_helm_string(polymer)
        assert scaffold in scaffold_designs, 'Scaffold %s not found in the scaffold designs.' % scaffold

        # Get the scaffold design for the current scaffold
        scaffold_design = scaffold_designs[scaffold]
        
        complex_polymer, connections, _, _ = parse_helm(polymer)

        # Generate mutants...
        for i in range(n):
            mutant_complex_polymer = {}
            n_mutations = 0

            for pid, simple_polymer in complex_polymer.items():
                mutated_simple_polymer = list(simple_polymer)

                # Residues involved in a connection within and between peptides won't be mutated
                if keep_connections and pid in complex_polymer.keys():
                    connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
                    connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
                    # -1, because positions are 1-based in HELM
                    connection_resids = np.asarray(connection_resids) - 1
                    possible_positions = list(set(range(len(simple_polymer))).difference(connection_resids))
                else:
                    possible_positions = list(range(len(simple_polymer)))

                # Choose a random number of mutations between min and max
                if minimum_mutations == maximum_mutations:
                    number_mutations = maximum_mutations
                elif maximum_mutations is None:
                    number_mutations = self._rng.integers(low=minimum_mutations, high=len(possible_positions))
                else:
                    # The maximum number of mutations cannot be greater than the length of the polymer
                    tmp_maximum_mutations = np.min([maximum_mutations, len(possible_positions)])
                    number_mutations = self._rng.integers(low=minimum_mutations, high=tmp_maximum_mutations)

                # Choose positions to mutate
                mutation_positions = self._rng.choice(possible_positions, size=number_mutations, replace=False)

                # Do mutations
                for mutation_position in mutation_positions:
                    # +1 , because positions are 1-based in HELM
                    chosen_monomer = self._rng.choice(scaffold_design[pid][mutation_position + 1])
                    mutated_simple_polymer[mutation_position] = chosen_monomer

                mutant_complex_polymer[pid] = (mutated_simple_polymer, mutation_positions)
                n_mutations += len(mutation_positions)

            if n_mutations > 0:
                if not keep_connections:
                    connections_to_keep = []

                    # Check if we have to remove connections due to the mutations
                    for i, connection in enumerate(connections):
                        # The connection positions must not be in the mutation lists
                        # mutant_polymers[connection['XXXXXPolymerID']][1] + 1 because positions are 1-based in HELM
                        if connection['SourceMonomerPosition'] not in mutant_complex_polymer[connection['SourcePolymerID']][1] + 1 and \
                           connection['TargetMonomerPosition'] not in mutant_complex_polymer[connection['TargetPolymerID']][1] + 1:
                            connections_to_keep.append(i)
                else:
                    connections_to_keep = list(range(connections.shape[0]))

                # Reconstruct the HELM string
                mutant_polymer = build_helm_string({p: s[0] for p, s in mutant_complex_polymer.items()}, connections[connections_to_keep])
                mutant_polymers.append(mutant_polymer)
            else:
                mutant_polymers.append(polymer)
        
        return mutant_polymers

    def crossover(self, polymer1, polymer2, cx_points=2):
        """
        Apply crossover genetic operator on the two polymers.

        Parameters
        ----------
        polymer1 : str
            The first parent polymer in HELM format.
        polymer2 : str
            The second parent polymer in HELM format.
        cx_points : int, default : 2
            The number of crossover points.

        Returns
        -------
        Tuple of str
            Two mutated polymers (children) in HELM format.

        Raises
        ------
        ValueError
            If polymers in the two input HELM strings have different simple polymer ids or 
            different lengths, or if the scaffold (connections) in the two input HELM strings 
            are different.

        """
        mutant_complex_polymer1 = {}
        mutant_complex_polymer2 = {}

        scaffold1 = get_scaffold_from_helm_string(polymer1)
        scaffold2 = get_scaffold_from_helm_string(polymer2)
        assert scaffold1 == scaffold2, f'Polymers must have the same scaffold ({scaffold1} != {scaffold2}).)'

        complex_polymer1, connections1, _, _ = parse_helm(polymer1)
        complex_polymer2, connections2, _, _ = parse_helm(polymer2)

        for pid in complex_polymer1.keys():
            # Copy parents since we are going to modify the children
            simple_polymer1 = list(complex_polymer1[pid])
            simple_polymer2 = list(complex_polymer2[pid])

            # Choose positions to crossover
            possible_positions = list(range(len(complex_polymer1[pid])))
            cx_positions = self._rng.choice(possible_positions, size=cx_points, replace=False)
            cx_positions = np.sort(cx_positions)

            for cx_position in cx_positions:
                simple_polymer1[cx_position:], simple_polymer2[cx_position:] = simple_polymer2[cx_position:], simple_polymer1[cx_position:]

            mutant_complex_polymer1[pid] = simple_polymer1
            mutant_complex_polymer2[pid] = simple_polymer2

        mutant_polymer1 = build_helm_string(mutant_complex_polymer1, connections1)
        mutant_polymer2 = build_helm_string(mutant_complex_polymer2, connections2)

        return (mutant_polymer1, mutant_polymer2)
