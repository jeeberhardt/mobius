#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Inverse folding
#

import esm
import numpy as np
import torch
import torch.nn.functional as F
from esm.inverse_folding.util import CoordBatchConverter


def _concatenate_chains(coords, seqs, target_chainid, padding_length=10):
    """
    Concatenates the coordinates and sequences of multiple chains, with padding in between, 
    and put the target chain first in concatenation.
   
    Parameters
    ----------
    coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
    seqs: Dictionary mapping chain ids to their sequences
    target_chainid: The chain id to put first in concatenation
    padding_length: Length of padding between concatenated chains

    Returns
    -------
    Tuple (coords, seq)
        - coords is an L x 3 x 3 array for N, CA, C coordinates, a
        concatenation of the chains with padding in between. The target chain 
        is put first in concatenation.
        - seq is a string representing the concatenated sequences of the chains
        with the target chain first in concatenation.
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)

    # For best performance, put the target chain first in concatenation.
    coords_list = [coords[target_chainid]]
    seqs_concatenated = seqs[target_chainid]

    for chain_id, chain_coords in coords.items():
        if chain_id != target_chainid:
            coords_list.extend([pad_coords, chain_coords])
            seqs_concatenated += seqs[chain_id]

    coords_concatenated = np.concatenate(coords_list, axis=0)

    return coords_concatenated, seqs_concatenated


class InverseFolding:

    def __init__(self, device=None):
        """
        Initializes the InverseFolding class. Uses the ESM-IF1 model to compute probabilities of
        amino acids at each position based on a protein structure.

        Parameters
        ----------
        device : str or torch.device, default : None
            Device on which to run the model. Per default, the device is set to 
            'cuda' if available, otherwise to 'cpu'.

        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = device
        self._standard_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        self._model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self._tokenizer = CoordBatchConverter(alphabet)

        # Get vocabulary
        self._vocab = alphabet.all_toks

        # Get idx of the natural amino acids in the pLM vocab, so we select only these
        # And we keep the same amino acid order for each pLM
        self._vocabulary_idx = np.array([self._vocab.index(aa) for aa in self._standard_amino_acids if aa in self._vocab])

        # Move model to device
        self._model.to(self._device)

        self._model.eval()

    @property
    def vocab(self):
        """Returns the tokens vocabulary."""
        return self._standard_amino_acids

    @property
    def model(self):
        """Returns the model."""
        return self._model

    @property
    def device(self):
        """Returns the device on which the model is running."""
        return self._device

    def _get_logits(self, coords, seqs):
        coords, confidence, _, tokens, padding_mask = self._tokenizer([(coords, None, seqs)], device=self.device)

        logits, _ = self._model.forward(coords, padding_mask, confidence, tokens)

        return logits

    def get_probabilities_from_structure(self, structure_filename, target_chainid, chainids=None, temperature=1.0):
        """
        Computes probabilities of amino acids at each position based on a protein structure.

        Parameters
        ----------
        structure_filename : str
            Path to either pdb or cif file
        target_chainid : str
            The chain id of the target from which the probabilities are obtained.
        chainids : list of str, default : None
            The chain id or list of chain ids to load. If None, all chains are loaded.
        temperature : float, default : 1.0
            The temperature parameter controls the sharpness of the probability distribution 
            for sequence sampling. Higher sampling temperatures yield more diverse sequences 
            but likely with lower native sequence recovery. To optimize for native sequence 
            recovery, a temperature as low as 1e-6 is recommended.

        Returns
        -------
        probabilities : torch.Tensor of shape (n_sequences, n_residues, n_amino_acids) or list of torch.Tensors of shape (n_residues, n_amino_acids)
            Probabilities of amino acids at each position per sequence. If return_probabilities is 
            True. Returns a list of torch.Tensors if the sequences have different lengths. The 
            probabilities for each residue follows this order: 'A', 'C', 'D', 'E', 'F', 'G', 'H', 
            'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'.

        """
        if chainids is not None:
            assert target_chainid in chainids, f'The target chain ID must be loaded (chain IDs loaded: {chainids})'

        structure = esm.inverse_folding.util.load_structure(structure_filename, chainids)
        coords, seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

        target_chain_len = coords[target_chainid].shape[0]
        all_coords, all_seqs = _concatenate_chains(coords, seqs, target_chainid)

        # Get logits
        logits = self._get_logits(all_coords, all_seqs)

        # Convert logits to probabilities
        logits = logits[0].transpose(0, 1)
        logits = logits[:, self._vocabulary_idx]
        probabilities = F.softmax(logits / temperature, dim=-1)

        # Get probabilities of the target chain only
        probabilities = probabilities[:target_chain_len]
        
        return probabilities.detach().cpu().numpy()

    @staticmethod
    def get_entropies_from_probabilities(probabilities):
        """
        Computes entropy from probabilities.

        Parameters
        ----------
        probabilities : numpy.ndarray of shape (n_residues, n_amino_acids)
            Probabilities of amino acids at each position.

        Returns
        -------
        entropy : ndarray of shape (n_residues,)
            Entropy at each position.
        """
        return -np.sum(probabilities * np.log(probabilities + 1e-10), axis=-1)
