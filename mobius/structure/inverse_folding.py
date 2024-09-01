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
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq is the extracted sequence, with padding tokens inserted
              between the concatenated chains
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)

    # For best performance, put the target chain first in concatenation.
    coords_list = [coords[target_chainid]]
    seqs_concatenated  = seqs[target_chainid]

    for chain_id in coords:
        if chain_id == target_chainid:
            continue

        coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])
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

        # Get special tokens
        self._mask_token = '<mask>'
        self._padding_idx = alphabet.padding_idx
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

    def _get_logits(self, coords, seq):
        batch = [(coords, None, seq)]
        coords, confidence, _, tokens, padding_mask = self._tokenizer(batch, device=self.device)

        logits, _ = self._model.forward(batch, padding_mask, confidence, tokens)

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

        probabilities = probabilities.detach().cpu().numpy()

        return probabilities

    @staticmethod
    def get_entropies_from_probabilities(probabilities):
        """
        Computes entropy from probabilities.

        Parameters
        ----------
        probabilities : torch.Tensor of shape (n_sequences, n_tokens, n_amino_acids)
            Probabilities of amino acids at each position per sequence.

        Returns
        -------
        entropy : ndarray of shape (n_sequences, n_tokens)
            Entropy at each position per sequence.

        """
        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
        entropy = entropy.detach().cpu().numpy()

        return entropy
