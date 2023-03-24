#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic Operators
#

import json
import os
import random

import numpy as np
import torch

from . import utils
from .descriptors import Map4Fingerprint
from .kernels import TanimotoSimilarityKernel
from .utils import build_helm_string, parse_helm


def constrained_sum_sample_pos(n, total):
    """
    Return a randomly chosen list of n positive integers summing to total.

    Parameters
    ----------
    n : int
        The number of positive integers to generate.
    total : int
        The total sum of the generated integers.

    Returns
    -------
    ndarray
        A 1D numpy array of n positive integers summing to total.

    Notes
    -----
    https://stackoverflow.com/questions/3589214

    """
    dividers = sorted(random.sample(range(1, total), n - 1))
    return np.array([a - b for a, b in zip(dividers + [total], [0] + dividers)])


def constrained_sum_sample_nonneg(n, total):
    """
    Return a randomly chosen list of n nonnegative integers summing to total.

    Parameters
    ----------
    n : int
        Number of nonnegative integers in the list.
    total : int
        The sum of the nonnegative integers in the list.

    Returns
    -------
    ndarray
        A 1D numpy array of n nonnegative integers summing to total.

    Notes
    -----
    https://stackoverflow.com/questions/3589214

    """
    return np.array([x - 1 for x in constrained_sum_sample_pos(n, total + n)])


class GeneticOperators:
    """
    A class for applying genetic operators on sequences in HELM format.

    """
    
    def __init__(self, monomer_symbols=None, HELMCoreLibrary=None, seed=None):
        """
        Initialize the GeneticOperators class.

        Parameters
        ----------
        monomer_symbols : List, default : None
            List of monomer symbols to use for the mutations. If not provided, 
            the 20 canonical amino acids will be used.
        HELMCoreLibrary: str, default : None
            File path to the HELMCore library JSON file. If not provided, the default 
            HELMCore library will be used.
        seed: int, default : None
            Seed value for the random number generator.

        """
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
        
        self._random_seed = seed
        self._rng = np.random.default_rng(self._random_seed)
        self._connections_dtype = [('SourcePolymerID', 'U20'), ('TargetPolymerID', 'U20'),
                                   ('SourceMonomerPosition', 'i4'), ('SourceAttachment', 'U2'),
                                   ('TargetMonomerPosition', 'i4'), ('TargetAttachment', 'U2')]
        
    def insert(self, input_sequence, n=10, only_terminus=False, maximum_size=None):
        """
        Apply the insertion genetic operator on the input sequence. The 
        insert genetic operator is only implemented for linear polymers.

        Parameters
        ----------
        input_sequence: str
            Input sequence in HELM format.
        n: int, default : 10
            Total number of sequences to generate.
        only_terminus: bool, default : False
            Whether to insert mutations only at the N or C terminus.
        maximum_size: int, de fault : None
            Maximum size of the new mutated sequences.

        Returns
        -------
        List of str
            Mutated sequences in HELM format.

        Raises
        ------
        NotImplementedError
            If the input sequence contains connections.

        """
        mutant_sequences = []
        
        polymers, connections, _, _ = parse_helm(input_sequence)

        if connections:
            raise NotImplementedError('Can only insert new monomers in linear polymers.')
        
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
                mutant_sequence = build_helm_string({p: s[0] for p, s in mutant_polymers.items()})
                mutant_sequences.append(mutant_sequence)
            else:
                mutant_sequences.append(input_sequence)
        
        return mutant_sequences

    def delete(self, helm_string, n=10, only_terminus=False, minimum_size=None):
        """
        Apply the delete genetic operator on the input sequence. The 
        delete genetic operator is only implemented for linear polymers.

        Parameters
        ----------
        helm_string: str
            Input sequence in HELM format.
        n: int, default : 10
            Total number of sequences to generate.
        only_terminus: bool, default : False
            Whether to delete mutations only at the N or C terminus.
        minimum_size: int, default : None
            Minimum size of the new mutated sequences.

        Returns
        -------
        List of str
            Mutated sequences in HELM format.

        Raises
        ------
        NotImplementedError
            If the input sequence contains connections.

        """
        mutant_sequences = []
        
        polymers, connections, _, _ = parse_helm(helm_string)

        if connections:
            raise NotImplementedError('Can only delete monomers in linear polymers.')
        
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
                mutant_sequence = build_helm_string({p: s[0] for p, s in mutant_polymers.items()})
                mutant_sequences.append(mutant_sequence)
            else:
                mutant_sequences.append(helm_string)
        
        return mutant_sequences

    def mutate(self, input_sequence, n=10, minimum_mutations=1, maximum_mutations=None, keep_connections=True):
        """
        Apply the mutation genetic operator on the input sequence.

        Parameters
        ----------
        input_sequence : str
            Input sequence in HELM format.
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
                    # Because positions are 1-based in HELM
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
                    monomer_symbol = mutated_sequence[mutation_position]
                    
                    # Force mutation!
                    while monomer_symbol == mutated_sequence[mutation_position]:                        
                        chosen_monomer = self._rng.choice(self._monomer_symbols)
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
