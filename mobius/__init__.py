#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius
#

from .forcefield import ForceField
from .virtual_target import VirtualTarget

from .acquisition_functions import ExpectedImprovement, ProbabilityOfImprovement
from .acquisition_functions import Greedy, RandomImprovement, UpperConfidenceBound
from .mobius import Mobius
from .sampler import PolymerSampler
from .descriptors import SequenceDescriptors
from .fingerprints import MHFingerprint, Map4Fingerprint, MorganFingerprint
from .genetic_algorithm import SequenceGA, ParallelSequenceGA, RandomGA
from .surrogate_models import GPModel, DummyModel, RFModel
from .generators import monomers_scanning, alanine_scanning, random_monomers_scanning, properties_scanning, scrumbled_scanning, homolog_scanning
from .genetic_operators import GeneticOperators
from .kernels import TanimotoSimilarityKernel
from .emulators import LinearPeptideEmulator, FindMe
from .plotting import plot_results
from .filters import PeptideSelfAggregationFilter, PeptideSolubilityFilter
from .utils import parse_helm, build_helm_string, get_scaffold_from_helm_string, read_pssm_file
from .utils import affinity_binding_to_energy, energy_to_affinity_binding, ic50_to_pic50, pic50_to_ic50
from .utils import generate_random_linear_polymers, generate_random_polymers_from_designs
from .utils import adjust_polymers_to_designs, check_polymers_with_designs
from .utils import convert_FASTA_to_HELM, convert_HELM_to_FASTA

__all__ = ['VirtualTarget', 'ForceField',
           'LinearPeptideEmulator', 'FindMe',
           'Mobius',
           'PolymerSampler',
           'GPModel', 'DummyModel', 'RFModel',
           'TanimotoSimilarityKernel',
           'ExpectedImprovement', 'ProbabilityOfImprovement', 'Greedy', 'RandomImprovement', 'UpperConfidenceBound',
           'MHFingerprint', 'Map4Fingerprint', 'MorganFingerprint', 'SequenceDescriptors',
           'GeneticOperators', 'SequenceGA', 'RandomGA', 'ParallelSequenceGA',
           'monomers_scanning', 'alanine_scanning', 'random_monomers_scanning', 'properties_scanning', 
           'scrumbled_scanning', 'homolog_scanning',
           'plot_results',
           'parse_helm', 'build_helm_string', 'get_scaffold_from_helm_string', 'read_pssm_file', 
           'affinity_binding_to_energy', 'energy_to_affinity_binding', 
           'generate_random_linear_polymers', 'generate_random_polymers_from_designs ',
           'adjust_polymers_to_designs', 'check_polymers_with_designs',
           'ic50_to_pic50', 'pic50_to_ic50',
           'PeptideSelfAggregationFilter', 'PeptideSolubilityFilter',
           'convert_FASTA_to_HELM', 'convert_HELM_to_FASTA']
