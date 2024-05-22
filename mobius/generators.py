#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Polymer generators
#

import copy
import itertools
import os
import random

import numpy as np
import pandas as pd

from . import utils


def homolog_scanning(polymer, substitution_matrix=None, input_type='helm', positions=None):
    """
    This function performs the homolog scanning method on a given polymer by mutating each 
    of its position by the chemically most similar monomers. 

    Parameters
    ----------
    polymer : str
        The input polymer, either in FASTA or HELM format.
    substitution_matrix : pd.DataFrame, optional
        The substitution matrix to use for the homolog scanning. The substitution matrix is a
        pandas DataFrame containing a square similarity matrix with indices and columns corresponding 
        to the monomer names (A, C, D, ... for standard amino acids). If not provided, it will 
        use the provided amino acid similarity matrix for the 20 standard amino acids (see 
        mobius/data/AA_similarity_matrix.csv).
    input_type : str, optional, default : 'helm'
        The format of the input polymer. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        The positions are 1-based. If not provided, all positions in the input polymer will 
        be scanned.

    Yields
    ------
    str
        The modified polymer in HELM format.

    Raises
    ------
    AssertionError
        If the input_type is not 'helm' or 'fasta'.
    KeyError
        If the monomer symbol is not present in the substitution matrix.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    i = 0

    # Parse the polymer based on input_type
    if input_type.lower() == 'helm':
        complex_polymer, connections, _, _ = utils.parse_helm(polymer)
    else:
        complex_polymer = {'PEPTIDE1': list(polymer)}
        connections = None

    # If substitution_matrix is not provided, use the default similarity matrix
    if substitution_matrix is None:
        d = utils.path_module("mobius")
        substitution_matrix_filename = os.path.join(d, "data/AA_similarity_matrix.csv")
        substitution_matrix = pd.read_csv(substitution_matrix_filename, index_col=0)

    allowed_positions = {}

    # Get allowed positions based on the input positions
    for pid, simple_polymer in complex_polymer.items():
        try:
            allowed_positions[pid] = np.array(positions[pid]) - 1
        except:
            allowed_positions[pid] = np.arange(0, len(simple_polymer))

        # Ignore positions involved into connections
        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    # Iterate through all the positions in the allowed positions
    for pid in itertools.cycle(complex_polymer.keys()):
        for position in allowed_positions[pid]:
            current_monomer = complex_polymer[pid][position]

            try:
                substitutions = substitution_matrix.loc[current_monomer].copy()
            except KeyError:
                error_msg = 'Error: Monomer symbol %s not present in the substitution matrix.' % current_monomer
                raise KeyError(error_msg)

            # Sort from the most similar to the least similar amino acid
            substitutions.sort_values(ascending=True, inplace=True)
            # Remove the current monomer from the list of substitutions
            substitutions.drop([current_monomer], inplace=True)

            try:
                new_monomer = substitutions.index.values[i]
            except IndexError:
                # It means we reach the end of all the possible substitutions
                break

            new_complex_polymer = copy.deepcopy(complex_polymer)
            new_complex_polymer[pid][position] = new_monomer
            new_polymer = utils.build_helm_string(new_complex_polymer, connections)

            if position == allowed_positions[pid][-1]:
                i += 1

            yield new_polymer
        else:
            continue

        break


def monomers_scanning(polymer, monomers=None, input_type='helm', positions=None):
    """
    This function performs a monomer scanning on a given polymer 
    by going through all the possible combinations of monomers at 
    each allowed position.
    
    Parameters
    ----------
    polymer : str
        The input polymer, either in FASTA or HELM format.
    monomers : list of str, default : None
        The list of allowed monomers. If not provided, the default list 
        of 20 amino acids is used.
    input_type : str, default : 'helm'
        The format of the input polymer. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        The positions are 1-based. If not provided, all positions in the input polymer will 
        be scanned.
    
    Yields
    ------
    str
        The modified polymer in HELM format.

    Raises
    ------
    AssertionError
        If the input_type is not 'helm' or 'fasta'.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    if monomers is None:
        # Default to the 20 natural amino acids
        monomers = ["A", "R", "N", "D", "C", "E", "Q", "G", 
                    "H", "I", "L", "K", "M", "F", "P", "S", 
                    "T", "W", "Y", "V"]

    if input_type.lower() == 'helm':
        # Parse the input polymer if in HELM format
        complex_polymer, connections, _, _ = utils.parse_helm(polymer)
    else:
        # Otherwise, assume a simple polymer and no connections
        complex_polymer = {'PEPTIDE1': list(polymer)}
        connections = None

    # Define the allowed positions for each polymer
    allowed_positions = {}

    for pid, simple_polymer in complex_polymer.items():
        try:
            allowed_positions[pid] = np.array(positions[pid]) - 1
        except:
            allowed_positions[pid] = np.arange(0, len(simple_polymer))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    for monomer in monomers:
        for pid in complex_polymer.keys():
            for position in allowed_positions[pid]:
                if complex_polymer[pid][position] != monomer:
                    new_complex_polymer = copy.deepcopy(complex_polymer)
                    new_complex_polymer[pid][position] = monomer
                    new_polymer = utils.build_helm_string(new_complex_polymer, connections)

                    yield new_polymer


def alanine_scanning(polymer, repeats=None, input_type='helm', positions=None):
    """
    This function performs alanine scanning on a given polymer 
    by introducing alanine at each position, and optionally by introducing 
    two or more alanine residues simultaneously.

    Parameters
    ----------
    polymer : str
        The input polymer, either in FASTA or HELM format.
    repeats : int or list-like, default : None
        The number of positions to introduce alanine at the same time. 
        If None, only introduce one alanine at a time. If int > 2, introduce that 
        many alanines at a time until all the possible combinations is exhausted. 
        If list-like, introduce that many alanines sequentially until all the possible 
        combinations is exhausted.
    input_type : str, default : 'helm'
        The format of the input polymer. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        The positions are 1-based. If not provided, all positions in the input polymer 
        will be scanned.

    Yields
    ------
    str
        The modified polymer in HELM format.

    Raises
    ------
    AssertionError
        If input_type is not 'fasta' or 'helm'.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    monomer = 'A'

    if repeats is not None:
        if not isinstance(repeats, (list, tuple, np.ndarray)):
            repeats = [repeats]
    else:
        repeats = []

    if input_type.lower() == 'helm':
        complex_polymer, connections, _, _ = utils.parse_helm(polymer)
    else:
        complex_polymer = {'PEPTIDE1': list(polymer)}
        connections = None

    allowed_positions = {}

    for pid, simple_polymer in complex_polymer.items():
        try:
            allowed_positions[pid] = np.array(positions[pid]) - 1
        except:
            allowed_positions[pid] = np.arange(0, len(simple_polymer))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    # Introduce only one Alanine at a time
    for pid in complex_polymer.keys():
        for position in allowed_positions[pid]:
            if complex_polymer[pid][position] != monomer:
                new_complex_polymer = copy.deepcopy(complex_polymer)
                new_complex_polymer[pid][position] = monomer
                new_polymer = utils.build_helm_string(new_complex_polymer, connections)

                yield new_polymer

    positions = [(pid, p) for pid, pos in allowed_positions.items() for p in pos]

    # Introduce two or more alanine at the same time if wanted
    for repeat in repeats:
        for repeat_positions in itertools.combinations(positions, repeat):
            new_complex_polymer = copy.deepcopy(complex_polymer)

            for position in repeat_positions:
                pid, i = position

                if complex_polymer[pid][i] != monomer:
                    new_complex_polymer[pid][i] = monomer

            new_polymer = utils.build_helm_string(new_complex_polymer, connections)

            if new_polymer != polymer:
                yield new_polymer


def random_monomers_scanning(polymer, monomers=None, input_type='helm', positions=None):
    """
    This function performs random monomers scanning method on a given 
    polymer by introducing random mutations at each position.

    Parameters
    ----------
    polymer : str
        The input polymer, either in FASTA or HELM format.
    monomers : List of str, default : None
        The list of allowed monomers. If not provided, the default list 
        of 20 amino acids is used.
    input_type : str, default : 'helm'
        The format of the input polymer. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        The positions are 1-based. If not provided, all positions in the input polymer 
        will be scanned.

    Yields
    ------
    str
        The modified polymer in HELM format.

    Raises
    ------
    AssertionError
        If the input type is not 'fasta' or 'helm'.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    if monomers is None:
        monomers = ["A", "R", "N", "D", "C", "E", "Q", "G", 
                    "H", "I", "L", "K", "M", "F", "P", "S", 
                    "T", "W", "Y", "V"]

    if input_type.lower() == 'helm':
        complex_polymer, connections, _, _ = utils.parse_helm(polymer)
    else:
        complex_polymer = {'PEPTIDE1': list(polymer)}
        connections = None

    allowed_positions = {}

    for pid, simple_polymer in complex_polymer.items():
        try:
            allowed_positions[pid] = np.array(positions[pid]) - 1
        except:
            allowed_positions[pid] = np.arange(0, len(simple_polymer))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    for pid in itertools.cycle(complex_polymer.keys()):
        for position in allowed_positions[pid]:
            monomer = np.random.choice(monomers)

            if complex_polymer[pid][position] != monomer:
                new_complex_polymer = copy.deepcopy(complex_polymer)
                new_complex_polymer[pid][position] = monomer
                new_polymer = utils.build_helm_string(new_complex_polymer, connections)

                yield new_polymer


def properties_scanning(polymer, properties=None, input_type='helm', positions=None):
    """
    This function performs a propety scanning on a given polymer 
    by introducing random monomer mutations while also alternating between 
    different properties. This ensures that each aminon acid properties are 
    better represented in the final polymer population.

    Parameters
    ----------
    polymer : str
        The input polymer, either in FASTA or HELM format.
    properties : Dict[str, List of str], default : None  
        A dictionary of properties to scan for, where keys are the name of the property, 
        and values are lists of amino acid codes that have that property. 
        Default is None, which will scan for the following properties:
    input_type : str, default : 'helm'
        The format of the input polymer. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        The positions are 1-based. If not provided, all positions in the input polymer 
        will be scanned.

    Returns
    -------
    str
        The modified polymer in HELM format.

    Raises
    ------
    AssertionError
        If the input_type is not 'helm' or 'fasta'.

    Examples
    --------
    >>> from mobius import properties_scanning
    >>> from mobius.utils import convert_FASTA_to_HELM
    >>> lead_peptide = convert_FASTA_to_HELM('HMTEVVRRC')[0]
    >>> properties = {
    …    'polar_pos' : ['R', 'H', 'K'],
    …    'polar_neg' : ['E', 'D'],
    …    'polar_neutral' : ['Q', 'T', 'G', 'C', 'N', 'S'],
    …    'polar_aro' : ['Y', 'W', 'F'],
    …    'polar_nonaro' : ['I', 'A', 'L', 'P', 'V', 'M']
    }
    >>> for peptide in properties_scanning(lead_peptide, properties=properties):
    …    print(peptide)

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    if properties is None:
        properties = {'polar_pos': ['R', 'H', 'K'],
                      'polar_neg': ['E', 'D'],
                      'polar_neutral': ['Q', 'T', 'G', 'C', 'N', 'S'],
                      'polar_aro': ['Y', 'W', 'F'],
                      'polar_nonaro': ['I', 'A', 'L', 'P', 'V', 'M']}

    if input_type.lower() == 'helm':
        complex_polymer, connections, _, _ = utils.parse_helm(polymer)
    else:
        complex_polymer = {'PEPTIDE1': list(polymer)}
        connections = None

    allowed_positions = {}

    for pid, simple_polymer in complex_polymer.items():
        try:
            allowed_positions[pid] = np.array(positions[pid]) - 1
        except:
            allowed_positions[pid] = np.arange(0, len(simple_polymer))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    for property_name in itertools.cycle(properties.keys()):
        for pid, simple_polymer in complex_polymer.items():
            for position in allowed_positions[pid]:
                monomer = np.random.choice(properties[property_name])

                if simple_polymer[position] != monomer:
                    new_complex_polymer = copy.deepcopy(complex_polymer)
                    new_complex_polymer[pid][position] = monomer
                    new_polymer = utils.build_helm_string(new_complex_polymer, connections)

                    yield new_polymer


def scrumbled_scanning(polymer, input_type='helm', positions=None):
    """
    This function performs a scrumbled scanning on a given polymer.
    Monomers in the input polymer are simply randomly shuffled.

    Parameters
    ----------
    polymer : str
        The input polymer, either in FASTA or HELM format.
    input_type : str, default : 'helm'
        The format of the input polymer. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        The positions are 1-based. If not provided, all positions in the input polymer 
        will be scanned.

    Yields
    ------
    str
        The modified polymer in HELM format.

    Raises
    ------
    AssertionError
        If the input_type is not 'helm' or 'fasta'.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    if input_type.lower() == 'helm':
        complex_polymer, connections, _, _ = utils.parse_helm(polymer)
    else:
        complex_polymer = {'PEPTIDE1': list(polymer)}
        connections = None

    allowed_positions = {}

    # For each polymer, get allowed positions or use all positions
    for pid, simple_polymer in complex_polymer.items():
        try:
            allowed_positions[pid] = np.array(positions[pid]) - 1
        except:
            allowed_positions[pid] = np.arange(0, len(simple_polymer))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    old_positions = [(pid, p) for pid, pos in allowed_positions.items() for p in pos]

    while True:
        new_positions = random.sample(old_positions, len(old_positions))
        new_complex_polymer = copy.deepcopy(complex_polymer)

        for new_position, old_position in zip(new_positions, old_positions):
            new_complex_polymer[new_position[0]][new_position[1]] = complex_polymer[old_position[0]][old_position[1]]

        new_polymer = utils.build_helm_string(new_complex_polymer, connections)

        yield new_polymer
