#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Peptide generators
#

import copy
import itertools
import os
import random

import numpy as np
import pandas as pd

from . import utils


class SubstitutionMatrix:
    """
    A class for managing substitution matrices.

    """

    def __init__(self, matrix_filename):
        """
        Initialize a SubstitutionMatrix object.

        Parameters
        ----------
        matrix_filename : str
            The name of the file containing the substitution matrix.

        """
        self._matrix = self._read_matrix(matrix_filename)

    def _read_matrix(self, matrix_filename):
        """
        Read a substitution matrix from a file.

        Parameters
        ----------
        matrix_filename : str
            The name of the file containing the substitution matrix.

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame representing the substitution matrix.

        """
        data = []
        columns = None

        with open(matrix_filename) as f:
            lines = f.readlines()

            for line in lines:
                if not line.startswith('#'):
                    if columns is None:
                        columns = [x.strip() for x in line.split(' ') if x]
                    else:
                        data.append([x.strip() for x in line.split(' ') if x][1:])

        data = np.array(data).astype(int)

        df = pd.DataFrame(data=data, columns=columns, index=columns)
        return df

    @property
    def shape(self):
        """
        Get the shape of the substitution matrix.

        Returns
        -------
        tuple
            A tuple representing the shape of the substitution matrix.

        """
        return self._matrix.shape

    def substitutes(self, monomer_symbol, include=False, ascending_order=False):
        """
        Get the substitution monomers for a given monomer symbol based on the substition matrix scoresÒ.

        Parameters
        ----------
        monomer_symbol : str
            The monomer symbol for which to get the substitution scores.
        include : bool, optional, default : False
            Whether to include the given monomer symbol in the output, by default False.
        ascending_order : bool, optional, default : False
            Whether to sort the output in ascending order, by default False.

        Returns
        -------
        ndarray
            An array of monomer symbols in the resquested order.

        """
        monomers_to_drop = ['X']

        try:
            substitutions = self._matrix.loc[monomer_symbol]
        except KeyError:
            error_msg = 'Error: Monomer symbol %s not present in the substitution matrix.' % monomer_symbol
            raise KeyError(error_msg)

        substitutions.sort_values(ascending=ascending_order, inplace=True)

        if not include:
            monomers_to_drop.append(monomer_symbol)

        substitutions.drop(monomers_to_drop, inplace=True)
        
        return substitutions.index.values


def homolog_scanning(input_sequence, substitution_matrix=None, input_type='helm', positions=None):
    """
    This function performs the homolog scanning method on a given peptide sequence 
    by mutating each position in the sequence by the chemically most similar monomers. 

    Parameters
    ----------
    input_sequence : str
        The input sequence, either in FASTA or HELM format.
    substitution_matrix : SubstitutionMatrix, optional
        The substitution matrix to use for scanning. If not provided, it will use the 
        default matrix 'VTML20.out'.
    input_type : str, optional, default : 'helm'
        The format of the input sequence. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        If not provided, all positions in the input sequence will be scanned.

    Yields
    ------
    str
        The modified sequence as a string in HELM format.

    Raises
    ------
    AssertionError
        If the input_type is not 'helm' or 'fasta'.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    i = 0

    # Parse the input_sequence based on input_type
    if input_type.lower() == 'helm':
        polymers, connections, _, _ = utils.parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    # If substitution_matrix is not provided, use the default matrix
    if substitution_matrix is None:
        d = utils.path_module("mobius")
        substitution_matrix_filename = os.path.join(d, "data/VTML20.out")
        substitution_matrix = SubstitutionMatrix(substitution_matrix_filename)

    allowed_positions = {}

    # Get allowed positions based on the input positions
    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        # Ignore positions involved into connections
        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    # Iterate through all the positions in the allowed positions
    for pid in itertools.cycle(polymers.keys()):
        for position in allowed_positions[pid]:
            try:
                monomer = substitution_matrix.substitutes(polymers[pid][position])[i]
            except IndexError:
                # It means we reach the end of all the possible substitutions
                break

            new_polymers = copy.deepcopy(polymers)
            new_polymers[pid][position] = monomer
            new_sequence = utils.build_helm_string(new_polymers, connections)

            if position == allowed_positions[pid][-1]:
                i += 1

            yield new_sequence
        else:
            continue

        break


def monomers_scanning(input_sequence, monomers=None, input_type='helm', positions=None):
    """
    This function performs a monomer scanning on a given peptide sequence 
    by going through all the possible combinations of monomers at 
    each allowed position.
    
    Parameters
    ----------
    input_sequence : str
        The input sequence, either in FASTA or HELM format.
    monomers : list of str, default : None
        The list of allowed monomers. If not provided, the default list 
        of 20 amino acids is used.
    input_type : str, default : 'helm'
        The format of the input sequence. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        If not provided, all positions in the input sequence will be scanned.
    
    Yields
    ------
    str
        The modified sequence as a string in HELM format.

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
        # Parse the input sequence if in HELM format
        polymers, connections, _, _ = utils.parse_helm(input_sequence)
    else:
        # Otherwise, assume a single peptide and no connections
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    # Define the allowed positions for each polymer
    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            # Use the provided positions, if any
            allowed_positions[pid] = positions[pid]
        except:
            # Otherwise, allow all positions
            allowed_positions[pid] = list(range(0, len(sequence)))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    for monomer in monomers:
        for pid in polymers.keys():
            for position in allowed_positions[pid]:
                if polymers[pid][position] != monomer:
                    new_polymers = copy.deepcopy(polymers)
                    new_polymers[pid][position] = monomer
                    new_sequence = utils.build_helm_string(new_polymers, connections)

                    yield new_sequence


def alanine_scanning(input_sequence, repeats=None, input_type='helm', positions=None):
    """
    This function performs alanine scanning on a given peptide sequence 
    by introducing alanine at each position, and optionally by introducing 
    two or more alanine residues simultaneously.

    Parameters
    ----------
    input_sequence : str
        The input sequence, either in FASTA or HELM format.
    repeats : int or list-like, default : None
        The number of positions to introduce alanine at the same time. 
        If None, only introduce one alanine at a time. If int > 2, introduce that 
        many alanines at a time until all the possible combinations is exhausted. 
        If list-like, introduce that many alanines sequentially until all the possible 
        combinations is exhausted.
    input_type : str, default : 'helm'
        The format of the input sequence. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        If not provided, all positions in the input sequence will be scanned.

    Yields
    ------
    str
        The modified sequence as a string in HELM format.

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
        polymers, connections, _, _ = utils.parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    # Introduce only one Alanine at a time
    for pid in polymers.keys():
        for position in allowed_positions[pid]:
            if polymers[pid][position] != monomer:
                new_polymers = copy.deepcopy(polymers)
                new_polymers[pid][position] = monomer
                new_sequence = utils.build_helm_string(new_polymers, connections)

                yield new_sequence

    positions = [(pid, p) for pid, pos in allowed_positions.items() for p in pos]

    # Introduce two or more alanine at the same time if wanted
    for repeat in repeats:
        for repeat_positions in itertools.combinations(positions, repeat):
            new_polymers = copy.deepcopy(polymers)

            for position in repeat_positions:
                pid, i = position

                if polymers[pid][i] != monomer:
                    new_polymers[pid][i] = monomer

            new_sequence = utils.build_helm_string(new_polymers, connections)

            if new_sequence != input_sequence:
                yield new_sequence


def random_monomers_scanning(input_sequence, monomers=None, input_type='helm', positions=None):
    """
    This function performs random monomers scanning method on a given 
    peptide sequence by introducing random mutations at each position.

    Parameters
    ----------
    input_sequence : str
        The input sequence, either in FASTA or HELM format.
    monomers : List of str, default : None
        The list of allowed monomers. If not provided, the default list 
        of 20 amino acids is used.
    input_type : str, default : 'helm'
        The format of the input sequence. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        If not provided, all positions in the input sequence will be scanned.

    Yields
    ------
    str
        The modified sequence as a string in HELM format.

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
        polymers, connections, _, _ = utils.parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    for pid in itertools.cycle(polymers.keys()):
        for position in allowed_positions[pid]:
            monomer = np.random.choice(monomers)

            if polymers[pid][position] != monomer:
                new_polymers = copy.deepcopy(polymers)
                new_polymers[pid][position] = monomer
                new_sequence = utils.build_helm_string(new_polymers, connections)

                yield new_sequence


def properties_scanning(input_sequence, properties=None, input_type='helm', positions=None):
    """
    This function performs a propety scanning on a given peptide sequence 
    by introducing random monomer mutations while also alternating between 
    different properties. This ensures that each aminon acid properties are 
    better represented in the final sequences.

    Parameters
    ----------
    input_sequence : str
        The input sequence, either in FASTA or HELM format.
    properties : Dict[str, List of str], default : None  
        A dictionary of properties to scan for, where keys are the name of the property, 
        and values are lists of amino acid codes that have that property. 
        Default is None, which will scan for the following properties:
    input_type : str, default : 'helm'
        The format of the input sequence. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        If not provided, all positions in the input sequence will be scanned.

    Returns
    -------
    str
        The modified sequence as a string in HELM format.

    Raises
    ------
    AssertionError
        If the input_type is not 'helm' or 'fasta'.

    Examples
    --------
    >>> from mobius import properties_scanning
    >>> from mobius.utils import convert_FASTA_to_HELM
    >>> input_sequence = convert_FASTA_to_HELM('HMTEVVRRC')[0]
    >>> properties = {
    …    'polar_pos' : ['R', 'H', 'K'],
    …    'polar_neg' : ['E', 'D'],
    …    'polar_neutral' : ['Q', 'T', 'G', 'C', 'N', 'S'],
    …    'polar_aro' : ['Y', 'W', 'F'],
    …    'polar_nonaro' : ['I', 'A', 'L', 'P', 'V', 'M']
    }
    >>> for sequence in properties_scanning(input_sequence, properties=properties):
    …    print(sequence)

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
        polymers, connections, _, _ = utils.parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if connections is not None:
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    for property_name in itertools.cycle(properties.keys()):
        for pid, polymer in polymers.items():
            for position in allowed_positions[pid]:
                monomer = np.random.choice(properties[property_name])

                if polymers[pid][position] != monomer:
                    new_polymers = copy.deepcopy(polymers)
                    new_polymers[pid][position] = monomer
                    new_sequence = utils.build_helm_string(new_polymers, connections)

                    yield new_sequence


def scrumbled_scanning(input_sequence, input_type='helm', positions=None):
    """
    This function performs a scrumbled scanning on a given peptide sequence.
    Monomers in the input sequence are simply randomly shuffled.

    Parameters
    ----------
    input_sequence : str
        The input sequence, either in FASTA or HELM format.
    input_type : str, default : 'helm'
        The format of the input sequence. Must be either 'fasta' or 'helm'. 
    positions : Dict[str, List of int], default : None
        The positions to be mutated, in the format {'polymer_id': [pos1, pos2, ...], ...}. 
        If not provided, all positions in the input sequence will be scanned.

    Yields
    ------
    str
        The modified sequence as a string in HELM format.

    Raises
    ------
    AssertionError
        If the input_type is not 'helm' or 'fasta'.

    """
    msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
    assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

    if input_type.lower() == 'helm':
        polymers, connections, _, _ = utils.parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    # For each polymer, get allowed positions or use all positions
    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

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
        new_polymers = copy.deepcopy(polymers)

        for new_position, old_position in zip(new_positions, old_positions):
            new_polymers[new_position[0]][new_position[1]] = polymers[old_position[0]][old_position[1]]

        new_sequence = utils.build_helm_string(new_polymers, connections)

        yield new_sequence
