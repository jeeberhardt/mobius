#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Serial BioPolymer Genetic algorithm
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
from pymoo.util.display.multi import MultiObjectiveOutput

from .display import SingleObjectiveOutput
from .terminations import NoChange
from .problem import Problem


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

            mutant_biopolymer1 = list(biopolymer1)
            mutant_biopolymer2 = list(biopolymer2)

            diff_positions = np.where(np.array(list(biopolymer1)) != np.array(list(biopolymer2)))[0]

            if diff_positions.size >= 2:
                # We don't want to do a crossever in parts where there are no differences
                # If there is just one difference or less (0), no need to do a crossover...
                possible_positions = list(range(diff_positions[0], diff_positions[-1] + 1))
                cx_positions = _rng.choice(possible_positions, size=self._cx_points, replace=False)
                cx_positions = np.sort(cx_positions)

                for cx_position in cx_positions:
                    mutant_biopolymer1[cx_position:], mutant_biopolymer2[cx_position:] = mutant_biopolymer2[cx_position:], mutant_biopolymer1[cx_position:]

            Y[0,k,0] = ''.join(mutant_biopolymer1)
            Y[1,k,0] = ''.join(mutant_biopolymer2)

        return Y


class BioPolymerMutation(Mutation):
    """
    Class to define mutation behaviour applied to new generation of biopolymers (in FASTA format).
    """

    def __init__(self, design, pm=0.1, minimum_mutations=1, maximum_mutations=None):
        """
        Initialize the mutation class for new generation of polymers.

        Parameters
        ----------
        design : dictionary
            Dictionnary of all the positions allowed to be optimized.
        pm : float, default : 0.1
            Probability of mutation.
        minimum_mutations : int, default : 1
            Minimal number of mutations introduced in the new child.
        maximum_mutations : int, default : None
            Maximal number of mutations introduced in the new child.

        """
        super().__init__()
        self._design = design
        self._pm = pm
        self._maximum_mutations = maximum_mutations
        self._minimum_mutations = minimum_mutations

    def _do(self, problem, X, **kwargs):
        _rng = np.random.default_rng()

        mutant_biopolymers = []

        possible_positions = [k for k, v in self._design.items() if v is not None]

        # for each individual
        for i in range(len(X)):
            r = _rng.random()

            # Applying mutation at defined probability rate
            if r < self._pm:
                biopolymer = X[i][0]
                mutant_biopolymer = list(biopolymer)

                # Choose a random number of mutations between min and max
                if len(possible_positions) == 1:
                    number_mutations = 1
                elif self._minimum_mutations == self._maximum_mutations:
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
                    monomers = self._design[mutation_position]['monomers']
                    probabilities = self._design[mutation_position]['probabilities']

                    # Choose a monomer based on the probabilities, provided by the user or None
                    chosen_monomer = _rng.choice(monomers, p=probabilities)

                    # -1 , because specific positions are 1-based in the design protocol
                    mutant_biopolymer[mutation_position - 1] = chosen_monomer

                mutant_biopolymer =  ''.join(mutant_biopolymer)
                mutant_biopolymers.append(mutant_biopolymer)
            else:
                mutant_biopolymers.append(X[i][0])

        mutant_biopolymers = np.array(mutant_biopolymers).reshape(-1, 1)

        return mutant_biopolymers


class DuplicateElimination(ElementwiseDuplicateElimination):
    """
    Class to prevent duplicate peptides being evaluated in a population.
    """

    def is_equal(self, a, b):
        return a.X[0] == b.X[0]


class SerialBioPolymerGA():
    """
    Class for the Single/Multi-Objectives GA optimization for biopolymers only.

    """

    def __init__(self, algorithm, n_gen=1000, n_pop=500, period=50, 
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=None,
                 save_history=False, **kwargs):
        """
        Initialize the Single/Multi-Objectives GA optimization for biopolymers only. The 
        SerialBioPolymerGA is not meant to be used directly. It is used by the
        SequenceGA class to run the optimization in parallel.

        Parameters
        ----------
        algorithm : str
            Algorithm to use for the optimization. Can be 'GA' for single-objective 
            optimization, or 'SMSEMOA', 'NSGA2' or 'AGEMOEA2' for multi-objectives optimization.
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

    def run(self, biopolymers, scores, acquisition_functions, design, filters=None):
        """
        Run the Single/Multi-Objectives GA optimization for biopolymers only.

        Parameters
        ----------
        biopolymers : array-like of str
            Biopolymers in FASTA format.
        scores : array-like of float or int
            Score associated to each biopolymer.
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the biopolymer.
        design : dictionnary
            Dictionnary of all the positions allowed to be optimized.
        filters : list of filter methods, default: None
            List of filter methods to use during the biopolymer optimization.

        Returns
        -------
        biopolymers : ndarray of shape (n_biopolymers,)
            Biopolymers found during the GA search in FASTA format.
        scores : ndarray of shape (n_biopolymers, n_scores)
            Score for each biopolymer found.

        """
        # Initialize the problem
        problem = Problem(biopolymers, scores, acquisition_functions, filters)

        # ... and pre-initialize the population with the experimental data.
        # This is only for the first GA generation.
        X = biopolymers.reshape(biopolymers.shape[0], -1)
        pop = Population.new("X", X)
        Evaluator().eval(problem, pop)

        # Turn off the pre-evaluation mode
        # Now it will use the acquisition scores from the surrogate models
        problem.eval()

        # Initialize genetic operators
        mutation = BioPolymerMutation(design, self._pm, self._minimum_mutations, self._maximum_mutations)
        crossover = BioPolymerCrossover(self._cx_points)
        duplicates = DuplicateElimination()

        # Initialize the GA method
        algorithm = self._method(pop_size=self._n_pop, sampling=pop, 
                                 crossover=crossover, mutation=mutation,
                                 eliminate_duplicates=duplicates)

        # Define termination criteria and make them robust to noise
        no_change_termination = RobustTermination(NoChange(), period=self._period)
        max_gen_termination = MaximumGenerationTermination(self._n_gen)
        termination = TerminateIfAny(max_gen_termination, no_change_termination)

        # Display function
        if self._optimization_type == 'single':
            output = SingleObjectiveOutput()
        else:
            output = MultiObjectiveOutput()

        # ... and run!
        results = minimize(problem, algorithm, termination=termination,
                           verbose=True, save_history=self._save_history, 
                           output=output)

        biopolymers = results.pop.get('X')
        scores = results.pop.get('F')

        return biopolymers, scores
