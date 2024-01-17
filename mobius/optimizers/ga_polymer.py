#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Serial Polymer Genetic algorithm
#

import warnings

import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.core.termination import TerminateIfAny 
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from pymoo.core.variable import get

from .terminations import NoChange
from .problem import Problem
from ..utils import adjust_polymers_to_designs
from ..utils import build_helm_string, parse_helm, get_scaffold_from_helm_string


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
        self._STRING_LENGTH_MULTIPLIER = 1.25

    def do(self, problem, pop, parents=None, **kwargs):

        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        # get the dimensions necessary to create in and output
        n_parents, n_offsprings = self.n_parents, self.n_offsprings
        n_matings, n_var = len(pop), problem.n_var

        # get the actual values from each of the parents
        X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1)
        if self.vtype is not None:
            X = X.astype(self.vtype)

        # Burger hack to avoid truncating the HELM strings
        if np.issubdtype(X.dtype, np.unicode_):
            current_length = int(X.dtype.str[2:])
            new_length = int(current_length * self._STRING_LENGTH_MULTIPLIER)
            dtype = f'U{new_length}'
        else:
            dtype = X.dtype

        # the array where the offsprings will be stored to
        Xp = np.empty(shape=(n_offsprings, n_matings, n_var), dtype=dtype)

        # the probability of executing the crossover
        prob = get(self.prob, size=n_matings)

        # a boolean mask when crossover is actually executed
        cross = np.random.random(n_matings) < prob

        # the design space from the parents used for the crossover
        if np.any(cross):

            # we can not prefilter for cross first, because there might be other variables using the same shape as X
            Q = self._do(problem, X, **kwargs)
            assert Q.shape == (n_offsprings, n_matings, problem.n_var), "Shape is incorrect of crossover impl."
            Xp[:, cross] = Q[:, cross]

        # now set the parents whenever NO crossover has been applied
        for k in np.flatnonzero(~cross):
            if n_offsprings < n_parents:
                s = np.random.choice(np.arange(self.n_parents), size=n_offsprings, replace=False)
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(np.random.permutation(n_parents))
                s = s[:n_offsprings]

            Xp[:, k] = np.copy(X[s, k])

        # flatten the array to become a 2d-array
        Xp = Xp.reshape(-1, X.shape[-1])

        # create a population object
        off = Population.new("X", Xp)

        return off

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

            Y[0,k,0] = mutant_polymer1
            Y[1,k,0] = mutant_polymer2

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
                polymer = X[i][0]

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


class SerialPolymerGA():
    """
    Class for the Single/Multi-Objectives GA optimization for polymers only.

    """

    def __init__(self, algorithm, designs=None, filters=None,
                 n_gen=1000, n_pop=500, period=50, cx_points=2, pm=0.1, 
                 minimum_mutations=1, maximum_mutations=None,
                 save_history=False, **kwargs):
        """
        Initialize the Single/Multi-Objectives GA optimization for polymers. The 
        SerialPolymerGA is not meant to be used directly. It is used by the
        SequenceGA class to run the optimization in parallel.

        Parameters
        ----------
        algorithm : str
            Algorithm to use for the optimization. Can be 'GA' for single-objective 
            optimization, or 'SMSEMOA', 'NSGA2' or 'AGEMOEA2' for multi-objectives optimization.
        designs : list of designs, default: None
            List of designs to use during the polymer optimization. If not provided,
            a default design protocol will be generated automatically based on the
            polymers provided during the optimization, with no filters.
        filters : list of filter methods, default: None
            List of filter methods to use during the polymer optimization.
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_population : int, default : 500
            Size of the population generated at each generation.
        period : int, default : 50
            Stopping criteria. Number of attempt before stopping the search. If no
            improvement is observed after `period` generations, we stop.
        cx_points : int, default : 2
            Number of crossing over during the mating step.
        pm : float, default : 0.1
            Probability of mutation.
        minimum_mutations : int, default : 1
            Minimal number of mutations introduced in the new child.
        maximum_mutations: int, default : None
            Maximal number of mutations introduced in the new child.
        save_history : bool, default : False
            Save the history of the optimization. This can be useful to debug the
            optimization, but it can take a lot of memory.

        """
        self._single = {'GA': GA}
        self._multi = {'NSGA2' : NSGA2, 'AGEMOEA2': AGEMOEA2, 'SMSEMOA': SMSEMOA}
        self._available_algorithms = self._single | self._multi

        msg_error = f'Only {list(self._available_algorithms.keys())} are supported, not {algorithm}'
        assert algorithm in self._available_algorithms, msg_error

        # Design protocol
        self._designs = designs
        self._filters = filters
        # GA Parameters
        self._optimization_type = 'single' if algorithm in self._single else 'multi'
        self._method = self._available_algorithms[algorithm]
        self._n_gen = n_gen
        self._n_pop = n_pop
        self._period = period
        self._cx_points = cx_points
        self._pm = pm
        self._minimum_mutations = minimum_mutations
        self._maximum_mutations = maximum_mutations
        self._save_history = save_history

    def run(self, polymers, scores, acquisition_functions):
        """
        Run the Single/Multi-Objectives GA optimization for polymers only.

        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the polymer.

        Returns
        -------
        polymers : ndarray of shape (n_polymers,)
            Polymers found during the GA search in HELM format.
        scores : ndarray of shape (n_polymers, n_scores)
            Score for each polymer found.

        """
        # Starts by automatically adjusting the input polymers to the design
        polymers, _ = adjust_polymers_to_designs(polymers, self._designs)

        # Initialize the problem
        problem = Problem(polymers, scores, acquisition_functions, self._filters)

        # ... and pre-initialize the population with the experimental data.
        # This is only for the first GA generation.
        X = polymers.reshape(polymers.shape[0], -1)
        pop = Population.new("X", X)
        Evaluator().eval(problem, pop)

        # Turn off the pre-evaluation mode
        # Now it will use the acquisition scores from the surrogate models
        problem.eval()

        # Initialize genetic operators
        mutation = PolymerMutation(self._designs, self._pm, self._minimum_mutations, self._maximum_mutations)
        crossover = PolymerCrossover(self._cx_points)
        duplicates = DuplicateElimination()

        # Initialize the GA method
        algorithm = self._method(pop_size=self._n_pop, sampling=pop, 
                                 crossover=crossover, mutation=mutation,
                                 eliminate_duplicates=duplicates)

        # Define termination criteria and make them robust to noise
        no_change_termination = RobustTermination(NoChange(), period=self._period)
        max_gen_termination = MaximumGenerationTermination(self._n_gen)
        termination = TerminateIfAny(max_gen_termination, no_change_termination)

        # ... and run!
        results = minimize(problem, algorithm, termination=termination,
                           verbose=True, save_history=self._save_history)

        polymers = results.pop.get('X')
        scores = results.pop.get('F')

        return polymers, scores
