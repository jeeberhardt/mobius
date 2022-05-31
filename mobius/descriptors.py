#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Descriptors
#

import numpy as np
from map4 import MAP4Calculator
from rdkit import Chem


class Map4Fingerprint:

    def __init__(self, input_type='fasta', dimensions=4096, radius=2, is_counted=False, is_folded=True):
        assert input_type.lower() in ['fasta', 'helm', 'smiles'], 'Format (%s) not handled. Please use FASTA, HELM or SMILES format.'

        self._map4calc = MAP4Calculator(dimensions=dimensions, radius=radius, is_counted=is_counted, is_folded=is_folded)
        self._input_type = input_type.lower()

    def transform(self, sequences):
        try:
            if self._input_type == 'fasta':
                mols = [Chem.rdmolfiles.MolFromFASTA(s) for s in sequences]
            elif self._input_type == 'helm':
                mols = [Chem.rdmolfiles.MolFromHELM(s) for s in sequences]
            else:
                mols = [Chem.rdmolfiles.MolFromSmiles(s) for s in sequences]
        except AttributeError:
            print('Error: there are issues with the input molecules')
            print(sequences)

        fps = self._map4calc.calculate_many(mols)
        fps = np.array(fps)

        return np.array(fps)


class SequenceDescriptors:

    def __init__(self, descriptors, input_type='helm'):
        self._descriptors = descriptors
        self._input_type = input_type

    def transform(self, sequences):    
        transformed = []

        for seq in sequences:
            tmp = []

            # The other input type is FASTA
            if self._input_type == 'helm':
                seq = ''.join(seq.split('$')[0].split('{')[1].split('}')[0].split('.'))

            for aa in seq:
                tmp.extend(self._descriptors[self._descriptors['AA1'] == aa].values[0][2:])

            transformed.append(tmp)

        return np.array(transformed)
