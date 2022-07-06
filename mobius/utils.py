#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - utils
#

from importlib import util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .helm import parse_helm, build_helm_string


def path_module(module_name):
    specs = util.find_spec(module_name)
    if specs is not None:
        return specs.submodule_search_locations[0]
    return None


def function_equal(func_1, func_2):
    return func_1.__code__.co_code == func_2.__code__.co_code


def opposite_signs(x, y):
    return ((x ^ y) < 0)


def affinity_binding_to_energy(value, unit='nM', temperature=300.):
    unit_converter = {'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1}
    RT = 0.001987 * temperature
    return RT * np.log(value * unit_converter[unit])


def energy_to_affinity_binding(value, unit='nM', temperature=300.):
    unit_converter = {'pM': 1e-12, 'nM': 1e9, 'uM': 1e6, 'mM': 1e3, 'M': 1}
    RT = 0.001987 * temperature
    return np.exp(value / RT) * unit_converter[unit]


def split_list_in_chunks(size, n):
    return [(l[0], l[-1]) for l in np.array_split(range(size), n)]


def generate_random_linear_peptides(n_peptides, peptide_lengths, monomer_symbols, output_format='helm'):
    random_peptides = []

    assert output_format.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    if not isinstance(peptide_lengths, (list, tuple)):
        peptide_lengths = [peptide_lengths]

    while True:
        peptide_length = np.random.choice(peptide_lengths)
        p = ''.join(np.random.choice(monomer_symbols, peptide_length))

        if output_format.lower() == 'helm':
            helm_string = build_helm_string({'PEPTIDE1': p}, [])
            random_peptides.append(helm_string)
        else:
            random_peptides.append(p)

        if len(random_peptides) == n_peptides:
            break

    return random_peptides
