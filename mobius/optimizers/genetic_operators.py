#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic Operators
#


import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from ..utils import build_helm_string, parse_helm, get_scaffold_from_helm_string


class BioPolymerCrossover(Crossover):
    """
    Class to define crossover behaviour for generating new generation of biopolymers (in FASTA format).
    """

    def __init__(self, cx_points=2):
        """
        Initialize the Crossover function.

        Parameters
        ----------
        cx_points : int, default : 2
            Number of crossing over during the mating step.

        Results
        ---------
        ndarray
            New generation of biopolymers (in FASTA format) from mating.

        """
        # define the crossover: number of parents and number of offsprings
        super().__init__(2,2)

        self._cx_points = cx_points

    def _do(self, problem, X, **kwargs):
        _rng = np.random.default_rng()

        # The input of has the following shape (n_parents, n_matings, n_var)
        offspring, n_matings, n_var = X.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            biopolymer1, biopolymer2 = X[0, k, 0], X[1, k, 0]

            msg_error = f'Biopolymers must have the same length: \n'
            msg_error += f'   ({len(biopolymer1)}: {biopolymer1} \n'
            msg_error += f'   ({len(biopolymer2)}: {biopolymer2} \n'
            assert len(biopolymer1) == len(biopolymer2), msg_error

            mutant_biopolymer1 = biopolymer1
            mutant_biopolymer2 = biopolymer2

            diff_positions = np.where(np.array(list(biopolymer1)) != np.array(list(biopolymer2)))[0]

            if diff_positions.size >= 2:
                # We don't want to do a crossever in parts where there are no differences
                # If there is just one difference or less (0), no need to do a crossover...
                possible_positions = list(range(diff_positions[0], diff_positions[-1] + 1))
                cx_positions = _rng.choice(possible_positions, size=self._cx_points, replace=False)
                cx_positions = np.sort(cx_positions)

                for cx_position in cx_positions:
                    mutant_biopolymer1[cx_position:], mutant_biopolymer1[cx_position:] = mutant_biopolymer2[cx_position:], mutant_biopolymer1[cx_position:]

            Y[0,k,0], Y[1,k,0] = mutant_biopolymer1, mutant_biopolymer2

        return Y


class BioPolymerMutation(Mutation):
    """
    Class to define mutation behaviour applied to new generation of biopolymers (in FASTA format).
    """

    def __init__(self, designs, pm=0.1, minimum_mutations=1, maximum_mutations=None):
        """
        Initialize the mutation class for new generation of polymers.

        Parameters
        ----------
        _designs : dictionary
            Dictionary with biopolymers (in FASTA format) and defined set of monomers to 
            use for each position.
        pm : float, default : 0.1
            Probability of mutation.
        minimum_mutations : int, default : 1
            Minimal number of mutations introduced in the new child.
        maximum_mutations : int, default : None
            Maximal number of mutations introduced in the new child.

        """
        super().__init__()
        self._designs = designs
        self._pm = pm
        self._maximum_mutations = maximum_mutations
        self._minimum_mutations = minimum_mutations

    def _do(self, problem, X, **kwargs):
        _rng = np.random.default_rng()

        mutant_biopolymers = []

        # for each individual
        for i in range(len(X)):
            r = _rng.random()

            # Applying mutation at defined probability rate
            if r < self._pm:
                biopolymer = X[i][0]
                mutant_biopolymer = list(biopolymer)
                possible_positions = list(self._designs['positions'].keys())

                # Choose a random number of mutations between min and max
                if self._minimum_mutations == self._maximum_mutations:
                    number_mutations = self._maximum_mutations
                elif self._maximum_mutations is None:
                    number_mutations = _rng.integers(low=self._minimum_mutations, high=len(possible_positions))
                else:
                    # The maximum number of mutations cannot be greater than the length of the polymer
                    tmp_maximum_mutations = np.min([self._maximum_mutations, len(possible_positions)])
                    number_mutations = _rng.integers(low=self._minimum_mutations, high=tmp_maximum_mutations)

                # Choose positions to mutate
                mutation_positions = _rng.choice(possible_positions, size=number_mutations, replace=False)

                # Do mutations
                for mutation_position in mutation_positions:
                    # +1 , because specific positions are 1-based in the design protocol
                    chosen_monomer = _rng.choice(self._design['positions'][mutation_position + 1])
                    mutant_biopolymer[mutation_position] = chosen_monomer

                mutant_biopolymer =  ''.join(mutant_biopolymer)
                mutant_biopolymers.append(mutant_biopolymer)
            else:
                mutant_biopolymers.append(X[i][0])

        mutant_biopolymers = np.array(mutant_biopolymers).reshape(-1, 1)

        return mutant_biopolymers


class PolymerCrossover(Crossover):
    """
    Class to define crossover behaviour for generating new generation of polymers (in HELM format).
    """

    def __init__(self, cx_points=2):
        """
        Initialize the Crossover function.

        Parameters
        ----------
        cx_points : int, default : 2
            Number of crossing over during the mating step.

        Results
        ---------
        ndarray
            New generation of polymers (in HELM format) from mating.

        """
        # define the crossover: number of parents and number of offsprings
        super().__init__(2,2)

        self._cx_points = cx_points

    def _do(self, problem, X, **kwargs):
        _rng = np.random.default_rng()

        # The input of has the following shape (n_parents, n_matings, n_var)
        offspring, n_matings, n_var = X.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            polymer1, polymer2 = X[0, k, 0], X[1, k, 0]

            #print("parents:",polymer1,polymer2)

            mutant_complex_polymer1 = {}
            mutant_complex_polymer2 = {}

            scaffold1 = get_scaffold_from_helm_string(polymer1)
            scaffold2 = get_scaffold_from_helm_string(polymer2)

            msg_error = f'Polymers must have the same scaffold: \n'
            msg_error += f'   ({scaffold1}: {polymer1} \n'
            msg_error += f'   ({scaffold2}: {polymer2} \n'
            assert scaffold1 == scaffold2, msg_error

            complex_polymer1, connections1, _, _ = parse_helm(polymer1)
            complex_polymer2, connections2, _, _ = parse_helm(polymer2)

            for pid in complex_polymer1.keys():
                # Copy parents since we are going to modify the children
                simple_polymer1 = list(complex_polymer1[pid])
                simple_polymer2 = list(complex_polymer2[pid])

                diff_positions = np.where(np.array(simple_polymer1) != np.array(simple_polymer2))[0]

                if diff_positions.size >= 2:
                    # We don't want to do a crossever in parts where there are no differences
                    # If there is just one difference or less (0), no need to do a crossover...
                    possible_positions = list(range(diff_positions[0], diff_positions[-1] + 1))
                    cx_positions = _rng.choice(possible_positions, size=self._cx_points, replace=False)
                    cx_positions = np.sort(cx_positions)

                    for cx_position in cx_positions:
                        simple_polymer1[cx_position:], simple_polymer2[cx_position:] = simple_polymer2[cx_position:], simple_polymer1[cx_position:]

                mutant_complex_polymer1[pid] = simple_polymer1
                mutant_complex_polymer2[pid] = simple_polymer2

            mutant_polymer1 = build_helm_string(mutant_complex_polymer1, connections1)
            mutant_polymer2 = build_helm_string(mutant_complex_polymer2, connections2)

            Y[0,k,0], Y[1,k,0] = mutant_polymer1, mutant_polymer2

        return Y


class PolymerMutation(Mutation):
    """
    Class to define mutation behaviour applied to new generation of polymers (in HELM format).
    """

    def __init__(self, scaffold_designs, pm=0.1, minimum_mutations=1, maximum_mutations=None, keep_connections=True):
        """
        Initialize the mutation class for new generation of polymers.

        Parameters
        ----------
        scaffold_designs : dictionary
            Dictionary with polymer scaffolds (in HELM format) and defined set of monomers to 
            use for each position.
        pm : float, default : 0.1
            Probability of mutation.
        minimum_mutations : int, default : 1
            Minimal number of mutations introduced in the new child.
        maximum_mutations : int, default : None
            Maximal number of mutations introduced in the new child.
        keep_connections : Bool, default : True
            Whether to retain connections between mutated residues and the rest of the polymer.

        """
        super().__init__()
        self._scaffold_designs = scaffold_designs
        self._pm = pm
        self._maximum_mutations = maximum_mutations
        self._minimum_mutations = minimum_mutations
        self._keep_connections = keep_connections

    def _do(self, problem, X, **kwargs):
        _rng = np.random.default_rng()

        mutant_polymers = []

        # for each individual
        for i in range(len(X)):
            r = _rng.random()

            # Applying mutation at defined probability rate
            if r < self._pm:
                polymer = X[i]
                polymer = polymer[0]

                scaffold = get_scaffold_from_helm_string(polymer)
                assert scaffold in self._scaffold_designs, 'Scaffold %s not found in the scaffold designs.' % scaffold

                scaffold_design = self._scaffold_designs[scaffold]

                complex_polymer, connections, _, _ = parse_helm(polymer)

                mutant_complex_polymer = {}
                n_mutations = 0

                for pid, simple_polymer in complex_polymer.items():
                    mutated_simple_polymer = list(simple_polymer)

                    # Residues involved in a connection within and between peptides won't be mutated
                    if self._keep_connections and pid in complex_polymer.keys():
                        connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
                        connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
                        # -1, because positions are 1-based in HELM
                        connection_resids = np.asarray(connection_resids) - 1
                        possible_positions = list(set(range(len(simple_polymer))).difference(connection_resids))
                    else:
                        possible_positions = list(range(len(simple_polymer)))

                    # Choose a random number of mutations between min and max
                    if self._minimum_mutations == self._maximum_mutations:
                        number_mutations = self._maximum_mutations
                    elif self._maximum_mutations is None:
                        number_mutations = _rng.integers(low=self._minimum_mutations, high=len(possible_positions))
                    else:
                        # The maximum number of mutations cannot be greater than the length of the polymer
                        tmp_maximum_mutations = np.min([self._maximum_mutations, len(possible_positions)])
                        number_mutations = _rng.integers(low=self._minimum_mutations, high=tmp_maximum_mutations)

                    # Choose positions to mutate
                    mutation_positions = _rng.choice(possible_positions, size=number_mutations, replace=False)

                # Do mutations
                    for mutation_position in mutation_positions:
                    # +1 , because positions are 1-based in HELM
                        chosen_monomer = _rng.choice(scaffold_design[pid][mutation_position + 1])
                        mutated_simple_polymer[mutation_position] = chosen_monomer

                mutant_complex_polymer[pid] = (mutated_simple_polymer, mutation_positions)
                n_mutations += len(mutation_positions)

                if n_mutations > 0:
                    if not self._keep_connections:
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
            else:
                polymer = X[i]
                polymer = polymer[0]
                mutant_polymers.append(polymer)

        mutant_polymers = np.array(mutant_polymers).reshape(-1, 1)

        return mutant_polymers


class DuplicateElimination(ElementwiseDuplicateElimination):
    """
    Class to prevent duplicate peptides being evaluated in a population.
    """

    def is_equal(self, a, b):
        return a.X[0] == b.X[0]
