#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - virtual target
#

import numpy as np
import pandas as pd


class VirtualTarget:
    """Class to handle a virtual target"""
    def __init__(self, forcefield, seed=None):
        """Initialize the virtual target


        Args:
            forcefield (Forcefield): a forcefield to score interaction between
                a sequence and the virtual target
            seed (int): random seed

        """
        self._forcefield = forcefield
        self._sequence_length = None
        self._pharmacophore = None
        self._random_seed = seed
        self._rng = np.random.default_rng(self._random_seed)

    def __repr__(self):
        repr_str = ""
        return repr_str

    def load_pharmacophore(self, input_filename):
        """Load the virtual phamacophore
        
        Args:
            input_filename (str): input csv filename containing the virtual
                pharmacophore

        """
        self._pharmacophore = pd.read_csv(input_filename)
        # Get the optimal sequence (sequence with the lowest score)
        self._optimal_sequence = self._find_optimal_sequence()

    def optimal_sequence(self):
        """Return the optimal sequence for the current pharmacophore
        
        Returns:
            str: optimal sequence

        """
        return self._optimal_sequence

    def _find_optimal_sequence(self):
        scores = []
        residue_names = self._forcefield.parameters()['AA1'].values

        # Score each residue at each position
        for residue_name in residue_names:
            sequence = [residue_name] * self._sequence_length
            _, score_details = self._forcefield.score(self._pharmacophore, sequence, True) 
            scores.append(score_details)

        # Get the index of the residues with the best (lowest) score
        residue_indices = np.argmin(scores, axis=0)
        # Get the residues names based on the index
        optimal_sequence = ''.join(residue_names[residue_indices])

        return optimal_sequence

    def generate_random_pharmacophore(self, sequence_length=None):
        """Generate a random pharmacophore

        Args:
            sequence_length (int): length of the sequence to generate

        """
        data = []
        self._sequence_length = sequence_length

        for i in range(self._sequence_length):
            solvent_exposure = self._rng.uniform()
            """The hydrophilicity of that position cannot be inferior than the solvent
            exposure. We want to avoid the not-so-realistic scenario where an hydrophobic
            residue is totally solvent exposed.
            Example:
                solvent_exposure = 0 --> hydrophilicity can be between 0 (hydrophobe) and 1 (polar)
                solvent_exposure = 1 --> Can only be 1 (polar)
            """
            hydrophilicity = self._rng.uniform(low=solvent_exposure)
            # In case the position is totally solvent exposed, the volume won't
            # have much effect in the scoring of that position
            volume = self._rng.uniform()
            data.append((solvent_exposure, hydrophilicity, volume))

        columns = ['solvent_exposure', 'hydrophilicity', 'volume']
        self._pharmacophore = pd.DataFrame(data=data, columns=columns)

        # Get the optimal sequence (sequence with the lowest score)
        self._optimal_sequence = self._find_optimal_sequence()

    def generate_pharmacophore_from_sequence(self, peptide_sequence, solvent_exposure=None):
        """Generate a pharmacophore from a peptide sequence
        
        Args:
            peptide_sequence (str): peptide sequence
            solvent_exposure (array-like): exposure to the solvent for each residue

        """
        assert len(peptide_sequence) == len(solvent_exposure), "The peptide sequence and solvent exposure have different size."

        parameters = 


    def generate_random_parent_peptides(self, n=1):
        """Generate peptide sequences that would not optimally 
        fit the pharmacophore

        Args:
            n (int): number of peptides to generate

        """
        parents = []
        return parents

    def score_peptides(self, peptides):
        """Score interaction between peptides and the virtual target
        using the provided forcefield

        Args:
            peptides (list): list of peptide strings

        Returns:
            np.ndarray: array of score for each peptide

        """
        score = []

        for p in peptides:
            score.append(self._forcefield.score(self._pharmacophore, p))

        return np.array(score)

    def export_pharmacophore(self, output_filename):
        """Export virtual pharmacophore as csv file

        Args:
            output_filename (str): output csv filename

        """
        self._pharmacophore.to_csv(output_filename, index=False)
