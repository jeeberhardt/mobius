#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Custom problem for MOO
#

import numpy as np
import torch
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from ..utils import get_scaffold_from_helm_string, parse_helm, build_helm_string
from ..planner import _load_design_from_config

class MyProblem(ElementwiseProblem):

    def __init__(self, acqs, n_var=1, n_obj=2, n_ieq_constr=0,greedy=False):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_ieq_constr=n_ieq_constr)
        
        self.polymer_cache = []
        
        assert len(acqs) == n_obj, "The number of acquisition functions is not equal to the number of objective functions."
        self.acqs = acqs

        if greedy == False:
            self.scaling = -1

        else:
            self.scaling = 1
            
    def _evaluate(self, x, out, *args, **kwargs):

        if self.polymer_cache is None or x not in [row[0] for row in self.polymer_cache]:
            scores = []

            for acq_fun in self.acqs:
                score = acq_fun.forward(x)
                ei = self.scaling*score.acq
                scores.append(ei[0])

            poly_track = [x[0]]

            for score in scores:
                poly_track.append(score)

            self.polymer_tracking(poly_track)

            out["F"] = np.asarray((scores),dtype=float)

        else:

            for i in range(len(self.polymer_cache)):
                if x == self.polymer_cache[i][0]:

                    scores = self.polymer_cache[i][1:]

                    out["F"] = np.array(scores,dtype=float)
                    break

    def get_acq_funs(self):

        return self.acqs

    def get_scaling(self):
        return self.scaling

    def polymer_tracking(self,scores):

        self.polymer_cache.append(scores)

    def get_polymer_cache(self):

        return self.polymer_cache

class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2,2)

    def _do(self, problem, X, **kwargs):

        _rng = np.random.default_rng()
        cx_points = 2
        # The input of has the following shape (n_parents, n_matings, n_var)
        offspring, n_matings, n_var = X.shape

        #print("matings",n_matings)
        #print("offspring",offspring)
        
        # The output owith the shape (n_offsprings, n_matings, n_var)
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
            assert scaffold1 == scaffold2, f'Polymers must have the same scaffold ({scaffold1} != {scaffold2}).)'
    
            complex_polymer1, connections1, _, _ = parse_helm(polymer1)
            complex_polymer2, connections2, _, _ = parse_helm(polymer2)

            for pid in complex_polymer1.keys():
                # Copy parents since we are going to modify the children
                simple_polymer1 = list(complex_polymer1[pid])
                simple_polymer2 = list(complex_polymer2[pid])
    
                # Choose positions to crossover
                possible_positions = list(range(len(complex_polymer1[pid])))
                cx_positions = _rng.choice(possible_positions, size=cx_points, replace=False)
                cx_positions = np.sort(cx_positions)
    
                for cx_position in cx_positions:
                    simple_polymer1[cx_position:], simple_polymer2[cx_position:] = simple_polymer2[cx_position:], simple_polymer1[cx_position:]
    
                mutant_complex_polymer1[pid] = simple_polymer1
                mutant_complex_polymer2[pid] = simple_polymer2

            mutant_polymer1 = build_helm_string(mutant_complex_polymer1, connections1)
            mutant_polymer2 = build_helm_string(mutant_complex_polymer2, connections2)
    
            Y[0,k,0], Y[1,k,0] = mutant_polymer1, mutant_polymer2
            
        return Y

class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):

        scaffold_designs = _load_design_from_config("design_protocol.yaml")

        _rng = np.random.default_rng()
        
        minimum_mutations = 1
        maximum_mutations = None
        keep_connections = True

        mutant_polymers = []

        #mutts_made = 0

        # for each individual
        for i in range(len(X)):

            r = _rng.random()
            #print("RNG Mutation Prob: ",r)

            if r < 0.1:

                #print("Randomly Mutated: r = ",r)
                #mutts_made += 1

                polymer = X[i]
                polymer = polymer[0]
                
                scaffold = get_scaffold_from_helm_string(polymer)
                assert scaffold in scaffold_designs, 'Scaffold %s not found in the scaffold designs.' % scaffold

                scaffold_design = scaffold_designs[scaffold]
            
                complex_polymer, connections, _, _ = parse_helm(polymer)
                
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
                        number_mutations = _rng.integers(low=minimum_mutations, high=len(possible_positions))
                    else:
                        # The maximum number of mutations cannot be greater than the length of the polymer
                        tmp_maximum_mutations = np.min([maximum_mutations, len(possible_positions)])
                        number_mutations = _rng.integers(low=minimum_mutations, high=tmp_maximum_mutations)

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

            else:

                polymer = X[i]
                polymer = polymer[0]
                mutant_polymers.append(polymer)


        #print("No. of mutants: ",mutts_made,"Total Polymers: ",len(X))

        mutant_polymers = np.array(mutant_polymers).reshape(-1, 1)

        return mutant_polymers

class MyDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.X[0] == b.X[0]
