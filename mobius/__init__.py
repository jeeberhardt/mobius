#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius
#

from .forcefield import ForceField
from .virtual_target import VirtualTarget

from .acquisition_functions import ExpectedImprovement, LogExpectedImprovement
from .acquisition_functions import RandomImprovement, LowerUpperConfidenceBound, ProbabilityOfImprovement
from .acquisition_functions import PosteriorMean, PosteriorStandardDeviation
from .mobius import Mobius
from .planner import Planner
from .descriptors import SimplePolymerDescriptors
from .fingerprints import MHFingerprint, Map4Fingerprint, MorganFingerprint
from .optimizers import SequenceGA, RandomGA
from .optimizers import Pool
from .surrogate_models import GPModel, GPLLModel, DummyModel, RFModel
from .embeddings import ProteinEmbedding
from .generators import monomers_scanning, alanine_scanning, random_monomers_scanning, properties_scanning, scrumbled_scanning, homolog_scanning
from .kernels import TanimotoSimilarityKernel, CosineSimilarityKernel
from .emulators import LinearPeptideEmulator, FindMe
from .rosetta import ProteinPeptideScorer
from .plotting import plot_results, visualise_2d, visualise_3d_scatter
from .filters import PeptideSelfAggregationFilter, PeptideSolubilityFilter
from .utils import parse_helm, build_helm_string, get_scaffold_from_helm_string, read_pssm_file
from .utils import affinity_binding_to_energy, energy_to_affinity_binding, ic50_to_pic50, pic50_to_ic50
from .utils import generate_random_linear_polymers, generate_random_polymers_from_designs
from .utils import adjust_polymers_to_designs, check_polymers_with_designs
from .utils import convert_FASTA_to_HELM, convert_HELM_to_FASTA,global_min_pssm_score
from .utils import optimisation_tracker


__all__ = ['VirtualTarget', 'ForceField',
           'LinearPeptideEmulator', 'FindMe',
           'Mobius',
           'Planner',
           'GPModel', 'GPLLModel', 'DummyModel', 'RFModel',
           'ProteinEmbedding',
           'TanimotoSimilarityKernel', 'CosineSimilarityKernel',
           'ExpectedImprovement', 'LogExpectedImprovement',
           'PosteriorMean', 'PosteriorStandardDeviation', 
           'ProbabilityOfImprovement', 'RandomImprovement', 'LowerUpperConfidenceBound',
           'MHFingerprint', 'Map4Fingerprint', 'MorganFingerprint', 'SimplePolymerDescriptors',
           'SequenceGA', 'RandomGA', 'Pool',
           'monomers_scanning', 'alanine_scanning', 'random_monomers_scanning', 'properties_scanning', 
           'scrumbled_scanning', 'homolog_scanning',
           'ProteinPeptideScorer',
           'plot_results',
           'parse_helm', 'build_helm_string', 'get_scaffold_from_helm_string', 'read_pssm_file', 
           'affinity_binding_to_energy', 'energy_to_affinity_binding', 
           'generate_random_linear_polymers', 'generate_random_polymers_from_designs',
           'adjust_polymers_to_designs', 'check_polymers_with_designs',
           'ic50_to_pic50', 'pic50_to_ic50',
           'PeptideSelfAggregationFilter', 'PeptideSolubilityFilter',
           'convert_FASTA_to_HELM', 'convert_HELM_to_FASTA','global_min_pssm_score','visualise_2d',
           'visualise_3d_scatter','optimisation_tracker']
