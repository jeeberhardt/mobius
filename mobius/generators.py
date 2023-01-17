#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Peptide generators
#

import itertools
import os
import random

import numpy as np

from .descriptors import SubstitutionMatrix
from . import utils


def homolog_scanning(input_sequence, substitution_matrix=None, input_type='helm', positions=None):
    assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    i = 0

    if input_type.lower() == 'helm':
        polymers, connections, _, _ = parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    if substitution_matrix is None:
        d = utils.path_module("mobius")
        substitution_matrix_filename = os.path.join(d, "data/VTML20.out")
        substitution_matrix = SubstitutionMatrix(substitution_matrix_filename)

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if not connections is None: 
            # Ignore positions involved into connections
            connection_resids = list(connections[connections['SourcePolymerID'] == pid]['SourceMonomerPosition'])
            connection_resids += list(connections[connections['TargetPolymerID'] == pid]['TargetMonomerPosition'])
            # Because positions are 1-based in HELM
            connection_resids = np.array(connection_resids) - 1
            allowed_positions[pid] = np.array(list(set(allowed_positions[pid]).difference(connection_resids)))

    for pid in itertools.cycle(polymers.keys()):
        for position in allowed_positions[pid]:
            try:
                monomer = substitution_matrix.substitutes(polymers[pid][position])[i]
            except IndexError:
                # It means we reach the end of all the possible substitutions
                break

            new_polymers = copy.deepcopy(polymers)
            new_polymers[pid][position] = monomer
            new_sequence = build_helm_string(new_polymers, connections)

            if position == allowed_positions[pid][-1]:
                i += 1

            yield new_sequence
        else:
            continue

        break


def monomers_scanning(input_sequence, monomers=None, input_type='helm', positions=None):
    assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    if monomers is None:
        monomers = ["A", "R", "N", "D", "C", "E", "Q", "G", 
                    "H", "I", "L", "K", "M", "F", "P", "S", 
                    "T", "W", "Y", "V"]

    if input_type.lower() == 'helm':
        polymers, connections, _, _ = parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if not connections is None: 
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
                    new_sequence = build_helm_string(new_polymers, connections)

                    yield new_sequence


def alanine_scanning(input_sequence, input_type='helm', positions=None, repeats=None):
    assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    monomer = 'A'

    if repeats is not None:
        if not isinstance(repeats, (list, tuple, np.ndarray)):
            repeats = [repeats]

    if input_type.lower() == 'helm':
        polymers, connections, _, _ = parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if not connections is None: 
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
                new_sequence = build_helm_string(new_polymers, connections)

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

            new_sequence = build_helm_string(new_polymers, connections)

            if new_sequence != input_sequence:
                yield new_sequence


def random_monomers_scanning(input_sequence, monomers=None, input_type='helm', positions=None):
    assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    if monomers is None:
        monomers = ["A", "R", "N", "D", "C", "E", "Q", "G", 
                    "H", "I", "L", "K", "M", "F", "P", "S", 
                    "T", "W", "Y", "V"]

    if input_type.lower() == 'helm':
        polymers, connections, _, _ = parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if not connections is None: 
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
                new_sequence = build_helm_string(new_polymers, connections)

                yield new_sequence


def properties_scanning(input_sequence, properties=None, input_type='helm', positions=None):
    assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    if properties is None:
        properties = {'polar_pos': ['R', 'H', 'K'],
                      'polar_neg': ['E', 'D'],
                      'polar_neutral': ['Q', 'T', 'G', 'C', 'N', 'S'],
                      'polar_aro': ['Y', 'W', 'F'],
                      'polar_nonaro': ['I', 'A', 'L', 'P', 'V', 'M']}

    if input_type.lower() == 'helm':
        polymers, connections, _, _ = parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if not connections is None:
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
                    new_sequence = build_helm_string(new_polymers, connections)

                    yield new_sequence


def scrumbled_scanning(input_sequence, input_type='helm'):
    assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

    if input_type.lower() == 'helm':
        polymers, connections, _, _ = parse_helm(input_sequence)
    else:
        polymers = {'PEPTIDE1': list(input_sequence)}
        connections = None

    allowed_positions = {}

    for pid, sequence in polymers.items():
        try:
            allowed_positions[pid] = positions[pid]
        except:
            allowed_positions[pid] = list(range(0, len(sequence)))

        if not connections is None:
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

        new_sequence = build_helm_string(new_polymers, connections)

        yield new_sequence
