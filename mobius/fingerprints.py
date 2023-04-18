#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Fingerprints
#

import itertools
from collections import defaultdict

import numpy as np
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetDistanceMatrix

from .utils import convert_HELM_to_FASTA, MolFromHELM


class MAP4Calculator:

    def __init__(self, dimensions=1024, radius=2, is_counted=False):
        """
        MAP4 calculator class
        """
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.encoder = MHFPEncoder()

    def calculate(self, mol):
        """Calculates the atom pair minhashed fingerprint
        Arguments:
            mol -- rdkit mol object
        Returns:
            tmap VectorUint -- minhashed fingerprint
        """
        
        atom_env_pairs = self._calculate(mol)
        return self._fold(atom_env_pairs)

    def calculate_many(self, mols):
        """ Calculates the atom pair minhashed fingerprint
        Arguments:
            mols -- list of mols
        Returns:
            list of tmap VectorUint -- minhashed fingerprints list
        """

        atom_env_pairs_list = [self._calculate(mol) for mol in mols]
        return [self._fold(pairs) for pairs in atom_env_pairs_list]

    def _calculate(self, mol):
        return self._all_pairs(mol, self._get_atom_envs(mol))

    def _fold(self, pairs):
        fp_hash = self.encoder.hash(set(pairs))
        return self.encoder.fold(fp_hash, self.dimensions)

    def _get_atom_envs(self, mol):
        atoms_env = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()

            for radius in range(1, self.radius + 1):
                if idx not in atoms_env:
                    atoms_env[idx] = []

                atoms_env[idx].append(MAP4Calculator._find_env(mol, idx, radius))

        return atoms_env

    @classmethod
    def _find_env(cls, mol, idx, radius):
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        atom_map = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)

        if idx in atom_map:
            smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
            return smiles

        return ''

    def _all_pairs(self, mol, atoms_env):
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)

        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = str(int(distance_matrix[idx1][idx2]))

            for i in range(self.radius):
                env_a = atoms_env[idx1][i]
                env_b = atoms_env[idx2][i]

                ordered = sorted([env_a, env_b])

                shingle = '{}|{}|{}'.format(ordered[0], dist, ordered[1])

                if self.is_counted:
                    shingle_dict[shingle] += 1
                    shingle += '|' + str(shingle_dict[shingle])

                atom_pairs.append(shingle.encode('utf-8'))

        return list(set(atom_pairs))


class Map4Fingerprint:
    """
    A class for computing the MAP4 fingerprints.

    """

    def __init__(self, input_type='helm_rdkit', dimensions=4096, radius=1, is_counted=False, HELMCoreLibrary_filename=None):
        """
        Constructs a new instance of the MAP4 fingerprint class.

        Parameters
        ----------
        input_type : str, optional (default='helm_rdkit')
            The input format for the sequences. It can be one of 'fasta', 
            'helm_rdkit', 'helm', or 'smiles'.
        dimensions : int, optional (default=4096)
            The length of the MAP4 fingerprint.
        radius : int, optional (default=1)
            The maximum radius of the subgraphs used for computing 
            the fingerprints.
        is_counted : bool, optional (default=False)
            If True, the function will return the number of atoms in 
            the subgraph instead of a binary vector.
        is_folded : bool, optional (default=True)
            If True, the fingerprint will be folded.
        HELMCoreLibrary_filename : str, optional (default=None)
            The path to the HELM Core Library file for HELM inputs.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA, HELM_rdkit, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm_rdkit', 'helm', 'smiles'], msg_error

        self._map4calc = MAP4Calculator(dimensions=dimensions, radius=radius, is_counted=is_counted)
        self._input_type = input_type.lower()
        self._HELMCoreLibrary_filename = HELMCoreLibrary_filename

    def transform(self, sequences):
        """
        Compute the MAP4 fingerprints for the given sequences.

        Parameters
        ----------
        sequences : str, list, tuple, or numpy.ndarray
            Input sequences to be transformed.

        Returns
        -------
        fingerprint : numpy.ndarray
            The MAP4 fingerprints for the input sequences.

        """
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


class MHFingerprint:
    """
    A class for computing the MHFingerprint.

    """

    def __init__(self, input_type='helm_rdkit', dimensions=4096, radius=3, rings=True, kekulize=True, HELMCoreLibrary_filename=None):
        """
        Constructs a new instance of the MHFingerprint class.

        Parameters
        ----------
        input_type : str, default 'helm_rdkit'
            Input format of the sequences. The options are 'fasta', 
            'helm_rdkit', 'helm', or 'smiles'.
        dimensions : int, default 4096
            The number of bits in the resulting binary feature vector.
        radius : int, default 3
            The radius parameter of the Morgan fingerprint.
        rings : bool, default True
            Whether to use information about the presence of rings 
            in the molecular structures.
        kekulize : bool, default True
            Whether to kekulize the structures before generating 
            the fingerprints.
        HELMCoreLibrary_filename : str or None, default None
            The file path to the HELM Core Library containing 
            the definitions of the monomers.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA, HELM_rdkit, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm_rdkit', 'helm', 'smiles'], msg_error

        self._dimensions = dimensions
        self._radius = radius
        self._rings = rings
        self._kekulize = kekulize
        self._encoder = MHFPEncoder()
        self._input_type = input_type.lower()
        self._HELMCoreLibrary_filename = HELMCoreLibrary_filename

    def transform(self, sequences):
        """
        Transforms the input sequences into MHFingerprints.

        Parameters
        ----------
        sequences : str, list or numpy.ndarray
            Input sequences to be transformed.

        Returns
        -------
        fingerprint : numpy.ndarray
           The MHFingerprint for the input sequences.

        """
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


class MorganFingerprint:
    """
    A class for computing Morgan Fingerprints.

    """

    def __init__(self, input_type='helm_rdkit', dimensions=4096, radius=2, HELMCoreLibrary_filename=None):
        """
        Constructs a new instance of the MorganFingerprint class.

        Parameters
        ----------
        input_type : str, default : 'helm_rdkit'
            The type of input sequences. It can be 'fasta', 'helm_rdkit', 
            'helm', or 'smiles'.
        dimensions : int, default : 4096
            The length of the fingerprint vector.
        radius : int, default : 2
            The radius of the Morgan fingerprint.
        HELMCoreLibrary_filename : str, default : None
            The filename of the HELMCore library. Only required if 
            input_type is 'helm' or 'helm_rdkit.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA, HELM_rdkit, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm_rdkit', 'helm', 'smiles'], msg_error

        self._radius = radius
        self._dimensions = dimensions
        self._input_type = input_type.lower()
        self._HELMCoreLibrary_filename = HELMCoreLibrary_filename

    def transform(self, sequences):
        """
        Transform input sequences into Morgan fingerprints.

        Parameters
        ----------
        sequences : list, tuple, ndarray or str
            The input sequences or a single sequence string.

        Returns
        -------
        fps : ndarray
            The Morgan fingerprints of the input sequences as a 2D numpy array.

        """
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
