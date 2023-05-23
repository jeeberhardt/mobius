#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - utils
#

import json
import os
import random
from importlib import util

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger


def constrained_sum_sample_pos(n, total):
    """
    Return a randomly chosen list of n positive integers summing to total.

    Parameters
    ----------
    n : int
        The number of positive integers to generate.
    total : int
        The total sum of the generated integers.

    Returns
    -------
    ndarray
        A 1D numpy array of n positive integers summing to total.

    Notes
    -----
    https://stackoverflow.com/questions/3589214

    """
    dividers = sorted(random.sample(range(1, total), n - 1))
    return np.array([a - b for a, b in zip(dividers + [total], [0] + dividers)])


def constrained_sum_sample_nonneg(n, total):
    """
    Return a randomly chosen list of n nonnegative integers summing to total.

    Parameters
    ----------
    n : int
        Number of nonnegative integers in the list.
    total : int
        The sum of the nonnegative integers in the list.

    Returns
    -------
    ndarray
        A 1D numpy array of n nonnegative integers summing to total.

    Notes
    -----
    https://stackoverflow.com/questions/3589214

    """
    return np.array([x - 1 for x in constrained_sum_sample_pos(n, total + n)])


def path_module(module_name):
    """
    Given a module name, return the path of the directory where the module is located.
    Returns None if the module does not exist.

    Parameters
    ----------
    module_name : str
        Name of the module.

    Returns
    -------
    path : str or None
        Path of the directory where the module is located, or None if the module does not exist.

    """
    specs = util.find_spec(module_name)
    if specs is not None:
        return specs.submodule_search_locations[0]
    return None


def function_equal(func_1, func_2):
    """
    Compare two functions to see if they are identical.

    Parameters
    ----------
    func_1 : function
        The first function to compare.
    func_2 : function
        The second function to compare.

    Returns
    -------
    bool
        True if the two functions are identical, False otherwise.

    """
    return func_1.__code__.co_code == func_2.__code__.co_code


def opposite_signs(x, y):
    """
    Return True if x and y have opposite signs, otherwise False.

    Parameters
    ----------
    x : float or int
        First number to compare.
    y : float or int
        Second number to compare.

    Returns
    -------
    bool
        True if x and y have opposite signs, False otherwise.

    """
    return ((x ^ y) < 0)


def affinity_binding_to_energy(value, unit='nM', temperature=300.):
    """
    Convert affinity binding to energy.

    Parameters
    ----------
    value : float
        Value of the affinity binding.
    unit : str, default: 'nM'
        Unit of the affinity binding.
    temperature : float, default: 300.
        Temperature at which to calculate the energy.

    Returns
    -------
    float
        Value of the energy corresponding to the given affinity binding.

    """
    unit_converter = {'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1}
    RT = 0.001987 * temperature
    return RT * np.log(value * unit_converter[unit])


def energy_to_affinity_binding(value, unit='nM', temperature=300.):
    """
    Convert energy to affinity binding.

    Parameters
    ----------
    value : float
        Value of the energy.
    unit : str, default: 'nM'
        Unit of the affinity binding.
    temperature : float, default: 300.
        Temperature at which to calculate the affinity binding.

    Returns
    -------
    float
        Value of the affinity binding corresponding to the given energy.

    """
    unit_converter = {'pM': 1e-12, 'nM': 1e9, 'uM': 1e6, 'mM': 1e3, 'M': 1}
    RT = 0.001987 * temperature
    return np.exp(value / RT) * unit_converter[unit]


def ic50_to_pic50(value, unit=None):
    """
    Convert IC50 to pIC50.

    Parameters
    ----------
    value : float
        Value of the IC50.
    unit : str or None, default: None
        Unit of the IC50.

    Returns
    -------
    float
        Value of the pIC50 corresponding to the given IC50.

    """
    unit_converter = {'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1, None: 1}
    return np.log10(value * unit_converter[unit])


def pic50_to_ic50(value, unit=None):
    """
    Converts a pIC50 value to IC50 value.

    Parameters
    ----------
    value : float
        The pIC50 value to be converted.
    unit : str, default : None
        The unit of the IC50 value.

    Returns
    -------
    float
        The IC50 value after the conversion.

    """
    unit_converter = {'pM': 1e12, 'nM': 1e9, 'uM': 1e6, 'mM': 1e3, 'M': 1, None: 1}
    return 10**value * unit_converter[unit]


def split(n, k):
    """
    Splits a number into k parts with each part as close to the same size as possible.

    Parameters
    ----------
    n : int
        The number to be split.
    k : int
        The number of parts to split the number into.

    Returns
    -------
    ndarray
        ndarray containing the parts of the split.

    """
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
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device


def generate_random_linear_polymers(n_polymers, polymers_lengths, monomers=None, output_format='helm'):
    """
    Generates random linear polymers.

    Parameters
    ----------
    n_polymers : int
        Number of random polymers to generate.
    polymers_lengths : List, tuple or numpy.ndarray
        List of polymers lengths to sample from.
    monomers : List of str, default : None
        A list of monomers to substitute at each allowed position. If not provided, 
        defaults to the 20 natural amino acids.
    output_format : str, default : 'helm'
        Output format. Can be 'fasta' or 'helm'.

    Returns
    -------
    List
        List of generated polymers.

    Raises
    ------
        AssertionError: If output format is not 'fasta' or 'helm'.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert output_format.lower() in ['fasta', 'helm'], msg_error

    if monomers is None:
        # Default to the 20 natural amino acids
        monomerss = ["A", "R", "N", "D", "C", "E", "Q", "G", 
                    "H", "I", "L", "K", "M", "F", "P", "S", 
                    "T", "W", "Y", "V"]

    if not isinstance(polymers_lengths, (list, tuple, np.ndarray)):
        polymers_lengths = [polymers_lengths]

    random_polymers = []

    while True:
        polymer_length = np.random.choice(polymers_lengths)
        p = ''.join(np.random.choice(monomers, polymer_length))

        if output_format.lower() == 'helm':
            helm_string = build_helm_string({'PEPTIDE1': p}, [])
            random_polymers.append(helm_string)
        else:
            random_polymers.append(p)

        if len(random_polymers) == n_polymers:
            break

    return random_polymers


def generate_random_polymers_from_designs(n_polymers, scaffold_designs):
    """
    Generates random polymers using scaffold_designs.

    Parameters
    ----------
    n_polymers : int
        Number of random polymers to generate.
    scaffold_designs : dictionary
        Dictionary with scaffold sequences and defined set of monomers 
        to use for each position.

    Returns
    -------
    List
        List of generated polymers.

    """
    random_polymers = []
    n_polymers_per_scaffold = constrained_sum_sample_nonneg(len(scaffold_designs), n_polymers)

    for scaffold, design in scaffold_designs.items():
        i = 0

        polymers, connections, _, _ = parse_helm(scaffold)

        for _ in range(n_polymers_per_scaffold[i]):
            mutant_polymers = {}

            for pid, sequence in polymers.items():
                mutated_sequence = list(sequence)

                for i, monomer in enumerate(sequence):
                    if monomer == 'X':
                        mutated_sequence[i] = np.random.choice(design[pid][i + 1])
                
                mutant_polymers[pid] = mutated_sequence
            
            mutant_sequence = build_helm_string(mutant_polymers, connections)
            random_polymers.append(mutant_sequence)

        i += 1

    return random_polymers


def check_polymers_with_designs(polymer_sequences, designs):
    """
    Checks if the given polymers are valid based on the designs.

    Parameters
    ----------
    polymer_sequences : List
        List of polymer sequences in HELM format.
    designs : dictionary
        Dictionary with scaffold sequences and defined set of monomers
        to use for each position.

    Returns
    -------
    ndarray
        ndarray of boolean values indicating whether the 
        polymers are valid or not based on the designs.

    """
    validity = np.ones(shape=(len(polymer_sequences),), dtype=bool)

    for i, polymer_sequence in enumerate(polymer_sequences):
        scaffold = get_scaffold_from_helm_string(polymer_sequence)

        try:
            scaffold_design = designs[scaffold]
        except:
            validity[i] = False
            continue

        polymers, _, _, _ = parse_helm(polymer_sequence)

        for pid, sequence in polymers.items():
            for i, monomer in enumerate(sequence):
                if monomer not in scaffold_design[pid][i + 1]:
                    validity[i] = False
                    break

    return validity


def adjust_polymers_to_designs(polymer_sequences, designs):
    """
    Modify polymers to fit the given designs.

    Parameters
    ----------
    polymer_sequences : List
        List of polymer sequences in HELM format.
    designs : dictionary
        Dictionary with scaffold sequences and defined set of monomers
        to use for each position.

    Returns
    -------
    List
        List of polymer sequences in HELM format.
    ndarray
        ndarray of boolean values indicating whether the 
        polymers was modified or not.

    """
    modified_sequences = []
    modified = np.zeros(shape=(len(polymer_sequences),), dtype=bool)

    for i, polymer_sequence in enumerate(polymer_sequences):
        scaffold = get_scaffold_from_helm_string(polymer_sequence)

        try:
            scaffold_design = designs[scaffold]
        except KeyError:
            msg_error = 'Scaffold %s corresponding to %s not found in the scaffold designs.' % (scaffold, polymer_sequence)
            raise KeyError(msg_error)

        sequences, connections, _, _ = parse_helm(polymer_sequence)

        for sid, sequence in sequences.items():
            modified_polymers = {}
            modified_sequence = list(sequence)

            for j, monomer in enumerate(sequence):
                if monomer not in scaffold_design[sid][j + 1]:
                    modified_sequence[j] = np.random.choice(scaffold_design[sid][j + 1])
                    modified[i] = True

            modified_polymers[sid] = modified_sequence

        modified_sequence = build_helm_string(modified_polymers, connections)
        modified_sequences.append(modified_sequence)

    return modified_sequences, modified


def convert_FASTA_to_HELM(fasta_sequences):
    """
    Converts one or more FASTA sequences to HELM format.

    Parameters
    ----------
    fasta_sequences : str, List of str, or ndarray of str
        A FASTA sequence or list/ndarray of FASTA sequences.

    Returns
    -------
    List of str
        A list of sequences in HELM format.

    """
    if not isinstance(fasta_sequences, (list, tuple, np.ndarray)):
        fasta_sequences = [fasta_sequences]

    return [build_helm_string({'PEPTIDE1': f}) for f in fasta_sequences]


def convert_HELM_to_FASTA(helm_sequences, ignore_connections=False):
    """
    Converts one or more HELM sequences to FASTA format.

    Parameters
    ----------
    helm_sequences : str, List of str, or numpy.ndarray of str
        A HELM sequence or list/array of HELM sequences.
    ignore_connections : bool, default : False
        Whether to ignore connections in HELM sequences.

    Returns
    -------
    List of str
        A list of sequences in FASTA format.

    Raises
    ------
    ValueError
        If a polymer contains connections or more than one sequence.

    """
    if not isinstance(helm_sequences, (list, tuple, np.ndarray)):
        helm_sequences = [helm_sequences]

    fasta_sequences = []

    for helm_sequence in helm_sequences:
        polymers, connections, _, _ = parse_helm(helm_sequence)

        if ignore_connections is False and connections:
            msg_error = 'Polymer %s cannot be converted to FASTA string. It contains connections.'
            raise ValueError(msg_error % helm_sequence)

        if len(polymers.keys()) > 1:
            msg_error = 'Polymer %s cannot be converted to FASTA string. It contains more than one sequence.'
            raise ValueError(msg_error % helm_sequence)

        fasta_sequences.append(''.join(polymers[list(polymers.keys())[0]]))

    return fasta_sequences


def build_helm_string(polymers, connections=None):
    """
    Build a HELM string from a dictionary of polymers and a list of connections.

    Parameters
    ----------
    polymers : dict
        A dictionary of polymers, where keys are the polymer types 
        and values are lists of monomer symbols.
    connections : List, default : None
        A list of connections, where each connection is represented 
        as a tuple with six elements: (start_polymer, start_monomer, start_attachment, 
        end_polymer, end_monomer, end_attachment).

    Returns
    -------
    str
        The generated HELM string.

    """
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
    """
    Parses a HELM string and returns the relevant information.

    Parameters
    ----------
    helm_string (str)
        A string in HELM format.

    Returns
    -------
    polymers : dict
        A dictionary with polymer IDs as keys and sequences as values.
    connections : numpy.ndarray
        An array with dtype [('SourcePolymerID', 'U20'), ('TargetPolymerID', 'U20'),\
                             ('SourceMonomerPosition', 'i4'), ('SourceAttachment', 'U2'),\
                             ('TargetMonomerPosition', 'i4'), ('TargetAttachment', 'U2')].
        Each row represents a connection between two monomers in the polymers.
    hydrogen_bonds : str
        A string containing information about any hydrogen bonds in the HELM string.
    attributes : str
        A string containing any additional attributes in the HELM string.

    """
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


def get_scaffold_from_helm_string(input_sequence):
    """
    Give the scaffold version of the HELM sequence.

    Parameters
    ----------
    input_sequence : str
        A sequence in HELM format.

    Returns
    -------
    str
        The scaffold version of the input sequence in HELM format.

    Examples
    --------
        input_sequence : PEPTIDE1{A.C.A.A.A}|PEPTIDE2{A.A.A.A}$PEPTIDE1,PEPTIDE2,1:R3-1:R3$$$V2.0
        scaffold       : PEPTIDE1{X.C.X.X.X}|PEPTIDE2{X.A.X.X}$PEPTIDE1,PEPTIDE2,1:R3-1:R3$$$V2.0

    """
    polymers, connections, _, _ = parse_helm(input_sequence)

    for polymer_id in polymers.keys():
        if connections.size > 0:
            # Get all the connections in this polymer
            attachment_positions1 = connections[connections['SourcePolymerID'] == polymer_id]['SourceMonomerPosition']
            attachment_positions2 = connections[connections['TargetPolymerID'] == polymer_id]['TargetMonomerPosition']
            attachment_positions = np.concatenate([attachment_positions1, attachment_positions2])
            # Build scaffold polymer sequence (X represents an unknown monomer in the HELM notation)
            scaffold_sequence = np.array(['X'] * len(polymers[polymer_id]))
            scaffold_sequence[attachment_positions - 1] = np.array(polymers[polymer_id])[attachment_positions - 1]
            # Replace polymer sequence by scaffold sequence
            polymers[polymer_id] = scaffold_sequence
        else:
            # Replace polymer sequence by scaffold sequence (but faster version since no connections)
            # (X represents an unknown monomer in the HELM notation)
            polymers[polymer_id] = 'X' * len(polymers[polymer_id])

    scaffold_sequence = build_helm_string(polymers, connections)

    return scaffold_sequence


def MolFromHELM(HELM_strings, HELMCoreLibrary_filename=None):
    """
    Generate a list of RDKit molecules from HELM strings.
    
    Parameters
    ----------
    HELM_strings : str or List or tuple or numpy.ndarray
        The input HELM string(s) to convert to RDKit molecules.
    HELMCoreLibrary_filename : str, default : None
        The filename of the HELM core library JSON file. 
        If not provided, the default HELMCoreLibrary JSON file will be used.
    
    Returns
    -------
    List
        A list of RDKit molecules.

    """
    peptides = []

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

        RDLogger.DisableLog('rdApp.warning')

        # Bop-it, Twist-it, Pull-it and Zip-it!
        peptide = Chem.molzip(rw_peptide)

        # Clean mol and remove dummy H atoms
        Chem.SanitizeMol(peptide)
        params = Chem.RemoveHsParameters()
        params.removeDegreeZero = True
        peptide = Chem.RemoveHs(peptide, params)

        RDLogger.EnableLog('rdApp.warning')

        peptides.append(peptide)

    return peptides


def read_pssm_file(pssm_file):
    """
    Reads a PSSM (position-specific scoring matrix) file and 
    returns a pandas DataFrame containing the data and the
    intercept value.

    Parameters
    ----------
    pssm_file : str
        The path to the PSSM file to be read.

    Returns
    -------
    pssm : pandas.DataFrame
        A DataFrame containing the data from the PSSM file.
    intercept : float
        The intercept value from the PSSM file.

    """
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
