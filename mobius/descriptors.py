#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Descriptors
#

import numpy as np
import pandas as pd
from map4 import MAP4Calculator
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import convert_HELM_to_FASTA, MolFromHELM


class MHFingerprint:

    def __init__(self, input_type='helm_rdkit', dimensions=4096, radius=3, rings=True, kekulize=True, HELMCoreLibrary_filename=None):
        assert input_type.lower() in ['fasta', 'helm_rdkit', 'helm', 'smiles'], 'Format (%s) not handled. Please use FASTA, HELM_rdkit, HELM or SMILES format.'

        self._dimensions = dimensions
        self._radius = radius
        self._rings = rings
        self._kekulize = kekulize
        self._encoder = MHFPEncoder()
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
                mols = MolFromHELM(sequences, self._HELMCoreLibrary_filename)
            else:
                mols = [Chem.rdmolfiles.MolFromSmiles(s) for s in sequences]
        except AttributeError:
            print('Error: there are issues with the input molecules')
            print(sequences)

        fps = [self._encoder.fold(self._encoder.encode_mol(m, radius=self._radius, rings=self._rings, kekulize=self._kekulize), length=self._dimensions) for m in mols]
        fps = np.asarray(fps)

        return fps


class Map4Fingerprint:

    def __init__(self, input_type='helm_rdkit', dimensions=4096, radius=1, is_counted=False, is_folded=True, HELMCoreLibrary_filename=None):
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
                mols = MolFromHELM(sequences, self._HELMCoreLibrary_filename)
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
                mols = MolFromHELM(sequences, self._HELMCoreLibrary_filename)
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
