#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic algorithm
#

import itertools
import os
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import ray

from .acquisition_functions import parallel_acq
from .helm import parse_helm, build_helm_string
from .helm_genetic_operators import HELMGeneticOperators
from .utils import generate_random_peptides, split_list_in_chunks


def _group_by_scaffold(helm_sequences, return_index=False):
    """Group helm strings by scaffold.

    Example:
        helm_sequence : PEPTIDE1{A.A.A.A.A}|PEPTIDE2{A.A.A.A}$PEPTIDE1,PEPTIDE2,1:R1-1:R1$$$V2.0
        scaffold      : PEPTIDE1{X.A.X.X.X}|PEPTIDE2{X.A.X.X}$PEPTIDE1,PEPTIDE2,1:R1-1:R1$$$V2.0

    """
    groups = defaultdict(list)
    group_indices = defaultdict(list)

    for i, helm_sequence in enumerate(helm_sequences):
        polymers, connections, _, _ = parse_helm(helm_sequence)

        for polymer_id in polymers.keys():
            if connections.size > 0:
                # Get all the connections in this polymer
                attachment_positions1 = connections[connections['SourcePolymerID'] == polymer_id]['SourceMonomerPosition']
                attachment_positions2 = connections[connections['TargetPolymerID'] == polymer_id]['TargetMonomerPosition']
                attachment_positions = np.concatenate([attachment_positions1, attachment_positions2])
                # Build scaffold polymer sequence (X represents an unknown monomer in the HELM notation)
                scaffold_sequence = np.array(['X'] * len(polymers[polymer_id]))
                scaffold_sequence[attachment_positions] = np.array(list(polymers[polymer_id]))[attachment_positions]
                # Replace polymer sequence by scaffold sequence
                polymers[polymer_id] = ''.join(scaffold_sequence)
            else:
                # Replace polymer sequence by scaffold sequence (but faster version since no connections)
                # (X represents an unknown monomer in the HELM notation)
                polymers[polymer_id] = 'X' * len(polymers[polymer_id])

        scaffold_sequence = build_helm_string(polymers, connections)

        groups[scaffold_sequence].append(helm_sequence)
        group_indices[scaffold_sequence].append(i)

    if return_index:
        return groups, group_indices
    else:
        return groups


def _boltzmann_probability(values, temperature=300.):
    p = np.exp(-(np.max(values) - values) / temperature)
    return p / np.sum(p)


def _number_of_children_to_generate_per_parent(parent_scores, n_children, temperature, greater_is_better=True):
    """Compute the number of children generated by each parent sequence 
    based on their acquisition score using Boltzmann weighting.
    """
    scaling_factor = (-1) ** (not greater_is_better)

    # Boltzmann weighting
    children_per_parent = np.floor(n_children * _boltzmann_probability(scaling_factor * parent_scores, temperature)).astype(int)

    current_n_children = np.sum(children_per_parent)

    # In the case, none of the parents are producing children
    # The best parent generate all the children
    if current_n_children == 0:
        i = np.argmax(scaling_factor * parent_scores)
        children_per_parent[i] = n_children

        return children_per_parent

    # We are going to add 1 until the number of children is equal to n_children
    # but only when the number of children is higher than zero   
    if current_n_children < n_children:
        # Get indices of parents that will have children and sort in descending order
        nonzero_parent_indices = np.argwhere(children_per_parent > 0).flatten()
        parent_indices = np.argsort(children_per_parent[nonzero_parent_indices])[::-1]

        children_per_parent[parent_indices[:np.sum(children_per_parent) - n_children]] += 1

    return children_per_parent


def _generate_mating_couples(parent_sequences, parent_scores, n_children, temperature, greater_is_better=True):
    """Generate mating couples based on their acquisition score of each parent 
    using Boltzmann weighting.
    """
    mating_couples = []
    n_couples = int(n_children / 2)
    parent_sequences = np.array(parent_sequences)

    scaling_factor = (-1) ** (not greater_is_better)

    try:
        p = _boltzmann_probability(scaling_factor * parent_scores, temperature)
    except ValueError:
        error_msg = 'parent_sequences : %s/nparent_scores    : %s' % (parent_sequences, parent_scores)
        raise ValueError(error_msg)

    mates_per_parent = np.floor(n_couples * p).astype(int)

    # In the case no parents really stood up from the rest, all
    # the <n_couples> best parents will be able to mate with someone
    if np.sum(mates_per_parent) == 0:
        print('Warning: It seems that none of the parents are worth mating. You might want to decrease the temperature.')
        i = np.argsort(scaling_factor * parent_scores)[::-1]
        mates_per_parent[i[:n_couples]] = 1

    # Complete to reach the number of couples asked
    if np.sum(mates_per_parent) < n_couples:
        nonzero_parent_indices = np.argwhere(mates_per_parent > 0).flatten()
        parent_indices = np.argsort(scaling_factor * parent_scores[nonzero_parent_indices])[::-1]

        for i in itertools.cycle(parent_indices):
            mates_per_parent[i] += 1

            if np.sum(mates_per_parent) == n_couples:
                break

    for idx in np.argwhere(mates_per_parent > 0).flatten():
        # Compute Boltzmann probabilities without the selected parent
        mask = np.ones(len(parent_sequences), dtype=bool)
        mask[idx] = False
        p = _boltzmann_probability(scaling_factor * parent_scores[mask], temperature)

        # Generate new couples with the selected parent
        mates = np.random.choice(parent_sequences[mask], size=mates_per_parent[idx], replace=True, p=p)
        mating_couples.extend([(parent_sequences[idx], m) for m in mates])

    return mating_couples


@ray.remote
def parallel_ga(gao, acquisition_function, sequences, scores):
    return gao.run(acquisition_function, sequences, scores)


class _GeneticAlgorithm(ABC):
    """Abstract class for genetic algorithm brick"""

    @abstractmethod
    def _generate_new_population(self, sequences, scores, greater_is_better):
        raise NotImplementedError()

    @abstractmethod
    def run(self, acquisition_function, sequences, scores=None):
        attempts = 0
        best_sequence_seen = None
        # Store all the sequences seen so far...
        sequences_cache = {}

        # Evaluate the initial population
        if scores is None:
            scores = acquisition_function.forward(sequences)

        for i in range(self._n_gen):
            # Generate new population
            sequences = self._generate_new_population(sequences, scores, acquisition_function.greater_is_better)

            # Keep only unseen sequences. We don't want to reevaluate known sequences...
            sequences_to_evaluate = list(set(sequences).difference(sequences_cache.keys()))

            # If there is no new sequences, we skip the evaluation
            if not sequences_to_evaluate:
                print('Warning: no new sequences were generated. Skip evaluation.')
                # But we still need to retrieve the scores of all the known sequences
                scores = np.array([sequences_cache[s] for s in sequences])
                continue

            # Evaluate new sequences
            sequences_to_evaluate_scores = acquisition_function.forward(sequences_to_evaluate)

            # Get scores of known sequences
            sequences_known = list(set(sequences).intersection(sequences_cache.keys()))
            sequences_known_scores = [sequences_cache[s] for s in sequences_known]

            # New population (known + unseen sequences)
            sequences = sequences_known + sequences_to_evaluate
            scores = np.concatenate([sequences_known_scores, sequences_to_evaluate_scores])

            # Store new sequences and scores in the cache
            sequences_cache.update(dict(zip(sequences_to_evaluate, sequences_to_evaluate_scores)))

            if acquisition_function.greater_is_better:
                idx = np.argmax(scores)
            else:
                idx = np.argmin(scores)

            current_best_sequence = sequences[idx]

            # Convergence criteria
            # If the best score does not improve after N attempts, we stop.
            if attempts < self._total_attempts:
                if best_sequence_seen == current_best_sequence:
                    attempts += 1
                else:
                    attempts = 0
                    best_sequence_seen = current_best_sequence
            else:
                print('Reached maximum number of attempts (%d), no improvement observed!' % self._total_attempts)
                break

            print('N %03d - Score: %.6f - Seq: %d - %s (%03d/%03d) - New seq: %d' % (i + 1, scores[idx], current_best_sequence.count('.'),
                                                                                     current_best_sequence, attempts, self._total_attempts,
                                                                                     len(sequences_to_evaluate)))

        all_sequences = np.array(list(sequences_cache.keys()))
        all_sequence_scores = np.fromiter(sequences_cache.values(), dtype=float)

        # Sort sequences by scores in the decreasing order (best to worst)
        if acquisition_function.greater_is_better:
            sorted_indices = np.argsort(all_sequence_scores)[::-1]
        else:
            sorted_indices = np.argsort(all_sequence_scores)

        self.sequences = all_sequences[sorted_indices]
        self.scores = all_sequence_scores[sorted_indices]

        return self.sequences, self.scores


class SequenceGA(_GeneticAlgorithm):

    def __init__(self, n_gen=1000, n_children=500, temperature=0.01, elitism=True, total_attempts=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=1, monomer_symbols=None,
                 **kwargs):
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_gen = n_gen
        self._n_children = n_children
        self._temperature = temperature
        self._elitism = elitism
        self._total_attempts = total_attempts
        self._helmgo = HELMGeneticOperators(monomer_symbols=monomer_symbols)
        # Parameters specific to SequenceGA
        self._cx_points = cx_points
        self._pm = pm
        self._minimum_mutations = minimum_mutations
        self._maximum_mutations = maximum_mutations

    def _generate_new_population(self, sequences, scores, greater_is_better):
        new_pop = []

        mating_couples = _generate_mating_couples(sequences, scores, self._n_children, self._temperature, greater_is_better)

        if self._elitism:
            # Carry-on the parents to the next generation
            new_pop.extend(list(np.unique(mating_couples)))

        for mating_couple in mating_couples:
            # This produces two children
            children = self._helmgo.crossover(mating_couple[0], mating_couple[1], self._cx_points)

            for child in children:
                if self._pm <= np.random.uniform():
                    child = self._helmgo.mutate(child, 1, self._minimum_mutations, self._maximum_mutations)[0]
                new_pop.append(child)

        return new_pop

    def run(self, acquisition_function, sequences, scores=None):
        _, group_indices = _group_by_scaffold(sequences, return_index=True)

        if len(group_indices) > 1:
            msg = 'presence of polymers with different scaffolds. Please use ParallelSequenceGA.'
            raise RuntimeError(msg)

        self.sequences, self.scores = super().run(acquisition_function, sequences, scores)

        print('End Scaffold GA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class ParallelSequenceGA(_GeneticAlgorithm):

    def __init__(self, n_gen=1000, n_children=500, temperature=0.01, elitism=True, total_attempts=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=1, monomer_symbols=None, 
                 n_process=-1, **kwargs):
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_process = n_process
        self._parameters = {'n_gen': n_gen, 'n_children': n_children, 
                            'temperature': temperature, 'elitism': elitism,
                            'total_attempts': total_attempts, 
                            'cx_points': cx_points, 'pm': pm,
                            'minimum_mutations': minimum_mutations, 
                            'maximum_mutations': maximum_mutations,
                            'monomer_symbols': monomer_symbols}
        self._parameters.update(kwargs)

    def _generate_new_population(self, sequences, scores, greater_is_better):
        raise NotImplementedError()

    def run(self, acquisition_function, sequences, scores=None):
        all_sequences = []
        all_sequence_scores = []

        # Evaluate the initial population
        if scores is None:
            scores = acquisition_function.forward(sequences)
        
        # Make sure that inputs are numpy arrays
        sequences = np.asarray(sequences)
        scores = np.asarray(scores)

        # Group/cluster peptides by scaffold
        _, group_indices = _group_by_scaffold(sequences, return_index=True)

        # Take the minimal amount of CPUs needed or available
        if self._n_process == -1:
            self._n_process = min([os.cpu_count(), len(group_indices)])

        # Dispatch all the scaffold accross different independent Sequence GA opt.
        ray.init(num_cpus=self._n_process, ignore_reinit_error=True)

        seq_gao = SequenceGA(**self._parameters)
        refs = [parallel_ga.remote(seq_gao, acquisition_function, sequences[seq_ids], scores[seq_ids]) for _, seq_ids in group_indices.items()]

        try:
            results = ray.get(refs)
        except KeyboardInterrupt:
            ray.shutdown()
            sys.exit(0)

        sequences, scores = zip(*results)
        ray.shutdown()

        # Remove duplicates
        all_sequences, unique_indices = np.unique(all_sequences, return_index=True)
        all_sequence_scores = np.array(all_sequence_scores)[unique_indices]

        # Sort sequences by scores in the decreasing order (best to worst)
        if acquisition_function.greater_is_better:
            sorted_indices = np.argsort(all_sequence_scores)[::-1]
        else:
            sorted_indices = np.argsort(all_sequence_scores)

        self.sequences = all_sequences[sorted_indices]
        self.scores = all_sequence_scores[sorted_indices]

        print('End SequenceGA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class ScaffoldGA(_GeneticAlgorithm):

    def __init__(self, n_gen=1, n_children=1000, temperature=0.1, elitism=True, total_attempts=50,
                 only_terminus=True, minimum_size=None, maximum_size=None, monomer_symbols=None,
                 **kwargs):
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_gen = n_gen
        self._n_children = n_children
        self._temperature = temperature
        self._elitism = elitism
        self._total_attempts = total_attempts
        self._helmgo = HELMGeneticOperators(monomer_symbols=monomer_symbols)
        # Parameters specific to ScaffoldGA
        self._only_terminus = only_terminus
        self._minimum_size = minimum_size
        self._maximum_size = maximum_size

    def _generate_new_population(self, sequences, scores, greater_is_better):
        new_pop = []

        # Compute the number of children generated by each parent sequence based on their acquisition score
        children_per_parent = _number_of_children_to_generate_per_parent(scores, self._n_children, self._temperature, greater_is_better)

        # Generate new population
        parent_indices = np.argwhere(children_per_parent > 0).flatten()

        for i in parent_indices:
            if self._elitism:
                # Carry-on the parents to the next generation
                new_pop.append(sequences[i])

            actions = np.random.choice(['insert', 'remove'], size=children_per_parent[i])
            new_pop.extend(self._helmgo.insert(sequences[i], np.sum(actions == "insert"), self._only_terminus, self._maximum_size))
            new_pop.extend(self._helmgo.delete(sequences[i], np.sum(actions == "remove"), self._only_terminus, self._minimum_size))

        return new_pop

    def run(self, acquisition_function, sequences, scores=None):
        self.sequences, self.scores = super().run(acquisition_function, sequences, scores)

        print('End Scaffold GA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class RandomGA():

    def __init__(self, n_process=-1, n_gen=1000, n_children=500, minimum_size=1, maximum_size=1, 
                 monomer_symbols=None, **kwargs):
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_process = n_process
        self._n_gen = n_gen
        self._n_children = n_children
        self._minimum_size = minimum_size
        self._maximum_size = maximum_size
        self._helmgo = HELMGeneticOperators(monomer_symbols=monomer_symbols)

    def run(self, acquisition_function, sequences=None, scores=None):
        all_sequences = []
        all_scores = []

        peptide_lengths = list(range(self._minimum_size, self._maximum_size + 1))

        chunks = split_list_in_chunks(self._n_children, self._n_process)

        for i in range(self._n_gen):
            sequences = generate_random_peptides(self._n_children, peptide_lengths, self._helmgo._monomer_symbols)
            refs = [parallel_acq.remote(acquisition_function, sequences[chunk[0]:chunk[1] + 1]) for chunk in chunks]
            scores = np.concatenate(ray.get(refs))

            all_sequences = np.append(all_sequences, sequences)
            all_scores = np.append(all_scores, scores)

        # Remove duplicates
        all_sequences, unique_indices = np.unique(all_sequences, return_index=True)
        all_scores = np.array(all_scores)[unique_indices]

        # Sort sequences by scores in the decreasing order (best to worst)
        if acquisition_function.greater_is_better:
            sorted_indices = np.argsort(all_scores)[::-1]
        else:
            sorted_indices = np.argsort(all_scores)

        self.sequences = all_sequences[sorted_indices]
        self.scores = all_scores[sorted_indices]

        print('End Random GA opt - Score: %5.3f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores
