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


def ic50_to_pic50(value, unit=None):
    unit_converter = {'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1, None: 1}
    return np.log10(value * unit_converter[unit])


def pic50_to_ic50(value, unit=None):
    unit_converter = {'pM': 1e12, 'nM': 1e9, 'uM': 1e6, 'mM': 1e3, 'M': 1, None: 1}
    return 10**value * unit_converter[unit]


def split(n, k):
    d, r = divmod(n, k)
    s = [d + 1] * r + [d] * (k - r)
    np.random.shuffle(s)
    return s


def split_list_in_chunks(size, n):
    return [(l[0], l[-1]) for l in np.array_split(range(size), n)]


def generate_random_linear_peptides(n_peptides, peptide_lengths, monomer_symbols, output_format='helm'):
    random_peptides = []

    assert output_format.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    if not isinstance(peptide_lengths, (list, tuple, np.ndarray)):
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


def convert_FASTA_to_HELM(fasta_sequences):
    if not isinstance(fasta_sequences, (list, tuple, np.ndarray)):
        fasta_sequences = [fasta_sequences]

    return [build_helm_string({'PEPTIDE1': f}) for f in fasta_sequences]


def convert_HELM_to_FASTA(helm_sequences):
    if not isinstance(helm_sequences, (list, tuple, np.ndarray)):
        helm_sequences = [helm_sequences]

    return [''.join(h.split('$')[0].split('{')[1].split('}')[0].split('.')) for h in helm_sequences]


def build_helm_string(polymers, connections=None):
    sequences_str = '|'.join(['%s{%s}' % (p, '.'.join(s)) for p, s in polymers.items()])
    if connections is not None:
        connections_str = '|'.join(['%s,%s,%d:%s-%d:%s' % (c[0], c[1], c[2], c[3], c[4], c[5]) for c in connections])
    else:
        connections_str = ''
    helm_string = '%s$%s$$$V2.0' % (sequences_str, connections_str)
    
    return helm_string


def parse_helm(helm_string):
    dtype = [('SourcePolymerID', 'U20'), ('TargetPolymerID', 'U20'),
             ('SourceMonomerPosition', 'i4'), ('SourceAttachment', 'U2'),
             ('TargetMonomerPosition', 'i4'), ('TargetAttachment', 'U2')]
    
    polymers, connections, hydrogen_bonds, attributes, _ = helm_string.split('$')
    
    # Process sequences
    data = {}
    for polymer in polymers.split('|'):
        pid = polymer.split('{')[0]
        sequence = polymer[len(pid) + 1:-1].split('.')
        data[pid] = sequence
        
    polymers = data
        
    # Process connections
    data = []
    if connections:
        for connection in connections.split('|'):
            source_id, target_id, con = connection.split(',')
            source_position, source_attachment = con.split('-')[0].split(':')
            target_position, target_attachment = con.split('-')[1].split(':')
            data.append((source_id, target_id,
                         source_position, source_attachment,
                         target_position, target_attachment))
        
    connections = np.array(data, dtype=dtype)
    
    return polymers, connections, hydrogen_bonds, attributes


def read_pssm_file(pssm_file):
    data = []
    intercept = np.nan
    AA = []

    with open(pssm_file) as f:
        lines = f.readlines()

        n_columns = int(lines[0].split('\t')[1])

        for line in lines[1:]:
            sline = line.strip().split('\t')

            if len(sline) == n_columns + 1:
                AA.append(sline[0])
                data.append([float(v) for v in sline[1:]])
            elif line.strip():
                intercept = float(sline[-1])

    columns = list(range(1, n_columns + 1))
    pssm = pd.DataFrame(data=data, columns=columns, index=AA)

    return pssm, intercept
