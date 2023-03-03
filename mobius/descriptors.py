#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Descriptors
#

import numpy as np
import pandas as pd
from map4 import MAP4Calculator
from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import convert_HELM_to_FASTA, MolFromHELM


class Map4Fingerprint:

    def __init__(self, input_type='helm_rdkit', dimensions=4096, radius=2, is_counted=False, is_folded=True, HELMCoreLibrary_filename=None):
        assert input_type.lower() in ['fasta', 'helm_rdkit', 'helm', 'smiles'], 'Format (%s) not handled. Please use FASTA, HELM_rdkit, HELM or SMILES format.'

        self._map4calc = MAP4Calculator(dimensions=dimensions, radius=radius, is_counted=is_counted, is_folded=is_folded)
        self._input_type = input_type.lower()
        self._HELMCoreLibrary_filename = HELMCoreLibrary_filename

    def transform(self, sequences):
        if not isinstance(sequences, (list, tuple, np.ndarray)):
            sequences = [sequences]

        try:
            if self._input_type == 'fasta':
                mols = [Chem.rdmolfiles.MolFromFASTA(s) for s in sequences]
            elif self._input_type == 'helm_rdkit':
                mols = [Chem.rdmolfiles.MolFromHELM(s) for s in sequences]
            elif self._input_type == 'helm':
                mols = [m for m in MolFromHELM(sequences, self._HELMCoreLibrary_filename)]
            else:
                mols = [Chem.rdmolfiles.MolFromSmiles(s) for s in sequences]
        except AttributeError:
            print('Error: there are issues with the input molecules')
            print(sequences)

        fps = self._map4calc.calculate_many(mols)
        fps = np.asarray(fps)

        return fps


class MorganFingerprint:

    def __init__(self, input_type='helm_rdkit', dimensions=4096, radius=2, HELMCoreLibrary_filename=None):
        assert input_type.lower() in ['fasta', 'helm_rdkit', 'helm', 'smiles'], 'Format (%s) not handled. Please use FASTA, HELM_rdkit, HELM or SMILES format.'

        self._radius = radius
        self._dimensions = dimensions
        self._input_type = input_type.lower()
        self._HELMCoreLibrary_filename = HELMCoreLibrary_filename

    def transform(self, sequences):
        if not isinstance(sequences, (list, tuple, np.ndarray)):
            sequences = [sequences]

        try:
            if self._input_type == 'fasta':
                mols = [Chem.rdmolfiles.MolFromFASTA(s) for s in sequences]
            elif self._input_type == 'helm_rdkit':
                mols = [Chem.rdmolfiles.MolFromHELM(s) for s in sequences]
            elif self._input_type == 'helm':
                mols = [m for m in MolFromHELM(sequences, self._HELMCoreLibrary_filename)]
            else:
                mols = [Chem.rdmolfiles.MolFromSmiles(s) for s in sequences]
        except AttributeError:
            print('Error: there are issues with the input molecules')
            print(sequences)

        GMFABV = AllChem.GetMorganFingerprintAsBitVect
        fps = [GMFABV(m, useChirality=True, useFeatures=True, radius=self._radius, nBits=self._dimensions) for m in mols]
        fps = np.asarray(fps)

        return fps


class SequenceDescriptors:

    def __init__(self, descriptors, input_type='helm'):
        assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

        self._descriptors = descriptors
        self._input_type = input_type

    def transform(self, sequences):    
        transformed = []

        # The other input type is FASTA
        if self._input_type == 'helm':
            sequences = convert_HELM_to_FASTA(sequences)

        for sequence in sequences:
            tmp = [self._descriptors[self._descriptors['AA1'] == aa].values[0][2:] for aa in sequence]
            transformed.append(tmp)

        return np.asarray(transformed)


class SubstitutionMatrix:

    def __init__(self, matrix_filename):
        self._matrix = self._read_matrix(matrix_filename)

    def _read_matrix(self, matrix_filename):
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
        return self._matrix.shape

    def substitutes(self, monomer_symbol, include=False, reverse_order=False):
        monomers_to_drop = ['X']

        try:
            substitutions = self._matrix.loc[monomer_symbol]
        except KeyError:
            error_msg = 'Error: Monomer symbol %s not present in the substitution matrix.' % monomer_symbol
            raise KeyError(error_msg)

        substitutions.sort_values(ascending=reverse_order, inplace=True)

        if not include:
            monomers_to_drop.append(monomer_symbol)

        substitutions.drop(monomers_to_drop, inplace=True)
        
        return substitutions.index.values
