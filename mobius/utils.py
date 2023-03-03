#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - utils
#

import json
import os
from importlib import util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem
from rdkit.Chem import molzip
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torhc.device("mps")
    else:
        device = torch.device("cpu")
    
    return device


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


def convert_HELM_to_FASTA(helm_sequences, ignore_connections=False):
    if not isinstance(helm_sequences, (list, tuple, np.ndarray)):
        helm_sequences = [helm_sequences]

    fasta_sequences = []

    for helm_sequence in helm_sequences:
        polymers, connections, _, _ = parse_helm(helm_sequence)

        if ignore_connections is False and connections:
            raise ValueError('Polymer %s cannot be converted to FASTA string. It contains connections.' % helm_sequence)

        if len(polymers.keys()) > 1:
            raise ValueError('Polymer %s cannot be converted to FASTA string. It contains more than one sequence.' % helm_sequence)

        fasta_sequences.append(''.join(polymers[list(polymers.keys())[0]]))

    return fasta_sequences


def build_helm_string(polymers, connections=None):
    sequences = []

    for p, s in polymers.items():
        tmp = '%s{%s}' % (p, '.'.join([m if len(m) == 1 else '[%s]' % m for m in s]))
        sequences.append(tmp)

    sequences_str = '|'.join(sequences)

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
        sequence = [monomer.strip("[]") for monomer in polymer[len(pid) + 1:-1].split('.')]
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


def MolFromHELM(HELM_strings, HELMCoreLibrary_filename=None):
    if not isinstance(HELM_strings, (list, tuple, np.ndarray)):
            HELM_strings = [HELM_strings]

    if HELMCoreLibrary_filename is None:
        d = path_module("mobius")
        HELMCoreLibrary_filename = os.path.join(d, "data/HELMCoreLibrary_new.json")

    with open(HELMCoreLibrary_filename) as f:
        data = json.load(f)
    
    # Re-organize monomer data in a dictionary for faster access
    HELMCoreLibrary = {monomer['symbol']: monomer for monomer in data}

    for HELM_string in HELM_strings:
        molecules_to_zip = []
    
        polymers, connections, _, _ = parse_helm(HELM_string)
        
        #print(polymers)
        #print(connections)
        
        for pid, seq in polymers.items():
            number_monomers = len(seq)

            for i, monomer_symbol in enumerate(seq):
                non_canonical_points = {}
                canonical_points = {}
                atom_attachments = {}
                
                try:
                    monomer_data = HELMCoreLibrary[monomer_symbol]
                except KeyError:
                    error_msg = 'Error: monomer %s unknown.' % monomer_symbol
                    print(error_msg)
                
                # Read SMILES string
                monomer = Chem.MolFromSmiles(monomer_data['smiles'])
                assert monomer is not None, 'Error: invalid monomer SMILES (%s) for %s' % (monomer_data['smiles'], monomer_symbol)
                
                #print(monomer_symbol, (i + 1), HELMCoreLibrary[monomer_symbol]['smiles'])
                
                # Get all the non-canonical attachment points
                if connections.size > 0:
                    c_ids = np.where((connections['SourceMonomerPosition'] == (i + 1)) | (connections['TargetMonomerPosition'] == (i + 1)))[0]
                    
                    if c_ids.size > 0:
                        for connection in connections[c_ids]:
                            if connection['SourcePolymerID'] == pid and connection['SourceMonomerPosition'] == (i + 1):
                                non_canonical_points['_%s' % connection['SourceAttachment']] = '_'.join(['%s' % s for s in connection])
                            elif connection['TargetPolymerID'] == pid and connection['TargetMonomerPosition'] == (i + 1):
                                non_canonical_points['_%s' % connection['TargetAttachment']] = '_'.join(['%s' % s for s in connection])
                    
                #print('Non-canonical attachments: ', non_canonical_points)
                
                # Get all the canonical attachment points (not involved in a non-canonical attachment)
                for r in monomer_data['rgroups']:
                    label = '_%s' % r['label']
                    
                    if label not in non_canonical_points:
                        monomer_cap = None
                        monomer_cap_smiles = None
                        
                        if label == '_R1':
                            if i == 0:
                                # If it is the first monomer and R1 is not involved in a non-canonical connections, then add cap
                                key = '%s_%d_%s' % (pid, (i + 1), r['label'])
                                monomer_cap_smiles = r['capGroupSmiles']
                            else:
                                # Look at R2 of the previous monomer and check if not involved in a non-canonical connections
                                r2_1 = np.where((connections['SourcePolymerID'] == pid) & (connections['SourceMonomerPosition'] == i) & (connections['SourceAttachment'] == 'R2'))[0]
                                r2_2 = np.where((connections['TargetPolymerID'] == pid) & (connections['TargetMonomerPosition'] == i) & (connections['TargetAttachment'] == 'R2'))[0]
                                 
                                if r2_1.size or r2_2.size:
                                    # R2_previous monomer involved in non-canonical connection 
                                    key = '%s_%d_%s' % (pid, (i + 1), r['label'])
                                    monomer_cap_smiles = r['capGroupSmiles']
                                    #print('R2 of previous monomer involved in a non-canonical attachments!!')
                                else:
                                    # Canonical connection between R2_previous and R1_current monomers
                                    key = '%s_%d_%d' % (pid, i, (i + 1))
                        elif label == '_R2':
                            if i == number_monomers - 1:
                                # If it is the last monomer and R2 is not involved in non-canonical connections, then add cap
                                key = '%s_%d_%s' % (pid, (i + 1), r['label'])
                                monomer_cap_smiles = r['capGroupSmiles']
                            else:
                                # Look at R1 of the next monomer and check if not involved in a non-canonical connections
                                r1_1 = np.where((connections['SourcePolymerID'] == pid) & (connections['SourceMonomerPosition'] == (i + 2)) & (connections['SourceAttachment'] == 'R1'))[0]
                                r1_2 = np.where((connections['TargetPolymerID'] == pid) & (connections['TargetMonomerPosition'] == (i + 2)) & (connections['TargetAttachment'] == 'R1'))[0]
                                
                                if r1_1.size or r1_2.size:
                                    # R1_next monomer involved in non-canonical connection 
                                    key = '%s_%d_%s' % (pid, (i + 1), r['label'])
                                    monomer_cap_smiles = r['capGroupSmiles']
                                    #print('R1 of next monomer involved in a non-canonical attachments!!')
                                else:
                                    # Canonical connection between R2_current and R2_next monomers
                                    key = '%s_%d_%d' % (pid, (i + 1), (i + 2))
                        else:
                            # For every R3, R4,... not involved in a non-canonical connections
                            key = '%s_%d_%s' % (pid, (i + 1), r['label'])
                            monomer_cap_smiles = r['capGroupSmiles']

                        canonical_points[label] = (key, monomer_cap_smiles)
                
                #print('Canonical attachments: ', canonical_points)
                
                # Set all the attachment points
                for atm in monomer.GetAtoms():
                    if atm.HasProp('atomLabel') and atm.GetProp('atomLabel').startswith('_R'):
                        label = atm.GetProp('atomLabel')
                        
                        if label in non_canonical_points:
                            # Set the non-canonical attachment points in the monomer
                            hashed_key = abs(hash(non_canonical_points[label])) % (10 ** 8)
                            atm.SetAtomMapNum(hashed_key)
                        elif label in canonical_points:
                            # Set the canonical attachment points in the monomer and monomer cap
                            hashed_key = abs(hash(canonical_points[label][0])) % (10 ** 8)
                            atm.SetAtomMapNum(hashed_key)
                            
                            # Set canonical attachment point in monomer cap
                            if canonical_points[label][1] is not None:
                                cap_smiles = canonical_points[label][1]
                                
                                cap = Chem.MolFromSmiles(cap_smiles)
                                assert cap is not None, 'Error: invalid monomer cap SMILES (%s) for %s' % (cap_smiles, monomer_symbol)
                                
                                for cap_atm in cap.GetAtoms():
                                    if cap_atm.HasProp('atomLabel') and cap_atm.GetProp('atomLabel') == label:
                                        #print('-- Monomer cap on: %s - %s (%d)' % (label, cap_smiles, hashed_key))
                                        cap_atm.SetAtomMapNum(hashed_key)
                                # ... and add monomer cap to peptide
                                molecules_to_zip.append(cap)
                        else:
                            print('Warning: attachment point %s not defined for monomer %s!' % (label, monomer_symbol))
                
                molecules_to_zip.append(monomer)
            
                #print('')
        
        with Chem.RWMol() as rw_peptide:
            [rw_peptide.InsertMol(molecule) for molecule in molecules_to_zip]
        
        # Bop-it, Twist-it, Pull-it and Zip-it!
        peptide = Chem.molzip(rw_peptide)
        
        # Clean mol and remove dummy H atoms
        Chem.SanitizeMol(peptide)
        params = Chem.RemoveHsParameters()
        params.removeDegreeZero = True
        peptide = Chem.RemoveHs(peptide, params)
        
        yield peptide


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
