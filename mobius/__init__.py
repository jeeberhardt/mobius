#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius
#

from .forcefield import ForceField
from .virtual_target import VirtualTarget

from .acquisition_functions import ExpectedImprovement, ProbabilityOfImprovement, Greedy, RandomImprovement
from .mobius import Mobius
from .sampler import PolymerSampler
from .descriptors import MHFingerprint, Map4Fingerprint, MorganFingerprint, SequenceDescriptors
from .ga import SequenceGA, ParallelSequenceGA, ScaffoldGA, RandomGA
from .surrogate_model import GPModel, DummyModel, RFModel
from .generators import monomers_scanning, alanine_scanning, random_monomers_scanning, properties_scanning, scrumbled_scanning, homolog_scanning, SubstitutionMatrix
from .helm_genetic_operators import HELMGeneticOperators
from .kernels import TanimotoSimilarityKernel
from .emulator import LinearPeptideEmulator, FindMe
from .plotting import plot_results
from .utils import parse_helm, build_helm_string, read_pssm_file
from .utils import affinity_binding_to_energy, energy_to_affinity_binding, generate_random_linear_peptides, ic50_to_pic50, pic50_to_ic50
from .utils import convert_FASTA_to_HELM, convert_HELM_to_FASTA

__all__ = ['VirtualTarget', 'ForceField',
           'LinearPeptideEmulator', 'FindMe',
           'Mobius',
           'PolymerSampler',
           'GPModel', 'DummyModel', 'RFModel',
           'TanimotoSimilarityKernel',
           'ExpectedImprovement', 'ProbabilityOfImprovement', 'Greedy', 'RandomImprovement',
           'MHFingerprint', 'Map4Fingerprint', 'MorganFingerprint', 'SequenceDescriptors',
           'HELMGeneticOperators', 'SequenceGA', 'ScaffoldGA', 'RandomGA', 'ParallelSequenceGA',
           'monomers_scanning', 'alanine_scanning', 'random_monomers_scanning', 'properties_scanning', 'scrumbled_scanning', 'homolog_scanning', 'SubstitutionMatrix',
           'plot_results',
           'parse_helm', 'build_helm_string', 'read_pssm_file', 
           'affinity_binding_to_energy', 'energy_to_affinity_binding', 'generate_random_linear_peptides', 'ic50_to_pic50', 'pic50_to_ic50',
           'onvert_FASTA_to_HELM', 'convert_HELM_to_FASTA']
