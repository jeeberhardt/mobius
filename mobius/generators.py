#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Peptide generators
#

import itertools

import numpy as np

from .helm import build_helm_string, parse_helm


def monomers_scanning(fasta_sequence, monomers, positions=None):
    if positions is None:
        positions = range(len(fasta_sequence))

    for position, monomer in itertools.product(positions, monomers):
        new_seq = fasta_sequence[:position] + monomer + fasta_sequence[position + 1:]

        if new_seq != fasta_sequence:
            yield new_seq
        else:
            pass


def alanine_scanning(fasta_sequence, positions=None, repeats=None):
    if positions is None:
        positions = range(len(fasta_sequence))

    if repeats is None:
        repeats = []

    for position, monomer in itertools.product(positions, ['A']):
        new_seq = fasta_sequence[:position] + monomer + fasta_sequence[position + 1:]

        if new_seq != fasta_sequence:
            yield new_seq
        else:
            pass

    for repeat in repeats:
        for repeat_positions in list(itertools.combinations(positions, repeat)):
            new_seq = (fasta_sequence + '.')[:-1]

            for position in repeat_positions:
                new_seq = new_seq[:position] + 'A' + new_seq[position + 1:]

            if new_seq != fasta_sequence:
                yield new_seq
            else:
                pass


def random_monomers_scanning(fasta_sequence, monomers, number_monomers_per_position=None, positions=None):
    if positions is None:
        positions = range(len(fasta_sequence))

    if number_monomers_per_position is None:
        number_monomers_per_position = len(monomers)

    for position in positions:
        for monomer in np.random.choice(monomers, size=number_monomers_per_position, replace=False):
            new_seq = fasta_sequence[:position] + monomer + fasta_sequence[position + 1:]

            if new_seq != fasta_sequence:
                yield new_seq
            else:
                pass


def properties_scanning(fasta_sequence, properties, positions=None):
    if positions is None:
        positions = range(len(fasta_sequence))

    for position in itertools.cycle(positions):
        monomers = [np.random.choice(v, size=1)[0] for _, v in properties.items()]

        for monomer in monomers:
            new_seq = fasta_sequence[:position] + monomer + fasta_sequence[position + 1:]

            if new_seq != fasta_sequence:
                yield new_seq
            else:
                pass
