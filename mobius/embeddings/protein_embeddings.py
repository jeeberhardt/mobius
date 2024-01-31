#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Protein Embeddings
#

import importlib

import esm
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .utils import replace_layers_with_lora, select_parameters
from .. import utils


class ProteinEmbedding:

    def __init__(self, pretrained_model_name='esm1b_t33_650M_UR50S', embedding_type='avg', 
                 device=None, model_name=None, tokenizer_name=None,
                 layers_to_finetune=None, lora=False, lora_rank=4, lora_alpha=8,
                 padding_length=None):
        """
        Initializes the ProteinEmbedding class.

        Parameters
        ----------
        pretrained_model_name : str
            Name of the encoder model. The model will be downloaded from the huggingface repository, 
            except for ESM models. Suggestions for pretrained_model_name are:
            - ESM models: esm1b_t33_650M_UR50S, esm2_t36_3B_UR50D
            - ProtT5: Rostlab/prot_t5_xl_uniref50, Rostlab/ProstT5, Rostlab/prot_bert
            - Others: TianlaiChen/PepMLM-650M
        embedding_type : str
            Type of embedding to use, either 'residue' or 'avg' (average). The number of output features 
            depends on the requested embedding type. With the 'avg' embedding type, the number of features 
            is equal to the out_features size of the model, while for the 'residue' embedding type, the 
            number of features will be equal to the out_features size times the number of tokens in the 
            sequence.
        device : str
            Device to use for embedding.
        device : str or torch.device, default : None
            Device on which to run the model. Per default, the device is set to 
            'cuda' if available, otherwise to 'cpu'.
        encoder_name : str, default : None
            Name of the encoder model to use if `AutoModel` failed to load the model.
        tokenizer_name : str, default : None
            Name of the tokenizer to use if `AutoTokenizer` failed to load the tokenizer.
        layers_to_finetune : str or List of str (default=None)
            Name the layers or weight parameters to finetune. Per default, no parameters are finetuned.
        lora : bool, default : False
            Whether to use LoRA layers. If True, the linear layers of the model will be replaced by
            LoRA layers. If False, apply traditional fine-tuning strategy (modify linear weights directly).
        lora_rank : int, default : 4
            Rank of the LoRA layer. Only if lora=True.
        lora_alpha : int, default : 8
            Scaling factor of the LoRA layer. Only if lora=True. As a rule of thumb, alpha is set to
            two times the rank (alpha = 2 * rank).
        padding_length : int, default : None
            Padding length for the tokenizer. Need to be specified if the SMILES of generated molecules 
            are longer than the SMILES used to train the surrogate model. This is used to ensure all 
            sequences have the same length. If None, the maximum length will be set to the length of 
            the longest sequence in the batch.

        Notes
        -----
        - Prot_t5_xl_uniref50: Use the `T5EncoderModel` model and `T5Tokenizer` tokenizer.
        - ProstT5: Use the `T5EncoderModel` model and `T5Tokenizer` tokenizer.

        Raises
        ------
        AssertionError
            If the embedding type is not supported.

        """
        assert embedding_type in ['residue', 'avg'], 'Only average (avg) and residue embeddings are supported.'

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = device
        self._pretrained_model_name = pretrained_model_name
        self._embedding_type = embedding_type
        self._standard_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self._layers_to_finetune = layers_to_finetune
        self._lora = lora
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._padding_length = padding_length

        if 'esm' in pretrained_model_name:
            self._model_type = 'esm'
            self._model, alphabet = esm.pretrained.load_model_and_alphabet(pretrained_model_name)
            self._tokenizer = alphabet.get_batch_converter()
            self._vocabulary_mask  = np.array([True if token in self._standard_amino_acids else False for token in alphabet.all_toks])
        else:
            self._model_type = 'other'

            if model_name is not None:
                module = importlib.import_module('transformers')
                model = getattr(module, model_name)
                self._model = model.from_pretrained(pretrained_model_name)
            else:
                self._model = AutoModel.from_pretrained(pretrained_model_name)

            if tokenizer_name is not None:
                module = importlib.import_module('transformers')
                tokenizer = getattr(module, tokenizer_name)
                self._tokenizer = tokenizer.from_pretrained(pretrained_model_name, legacy=False)
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, legacy=False)

            self._vocabulary_mask = None

        # Freeze all parameters
        self.freeze()

        # Setup model for finetuning
        if self._layers_to_finetune is not None:
            # Either replace layers with LoRA or apply traditional fine-tuning strategy 
            # by unfreezing weights in the selected layers.
            if self._lora:
                replace_layers_with_lora(self._model, self._layers_to_finetune, self._lora_rank, self._lora_alpha)
            else:
                self.unfreeze(self._layers_to_finetune)

        # Move model to device
        self._model.to(self._device)

        self.eval()

    @property
    def model(self):
        """Returns the model."""
        return self._model

    @property
    def device(self):
        """Returns the device on which the model is running."""
        return self._device

    def eval(self):
        """Sets the model to evaluation mode."""
        self._model.eval()

    def train(self):
        """Sets the model to training mode."""
        self._model.train()

    def freeze(self):
        """Freezes all parameters of the model."""
        for param in self._model.parameters():
            param.requires_grad = False

    def unfreeze(self, layers_to_unfreeze=None):
        """Unfreezes parameters of the model.

        Parameters
        ----------
        layers_to_unfreeze : str or List of str, default : None
            Name(s) or regular expression(s) to select the layers or weight parameters 
            to be unfrozen. If None, all parameters are unfreezed (".*"). 

        Raises
        ------
        ValueError
            If no parameters were found with the pattern.

        """
        if layers_to_unfreeze is None:
            layers_to_unfreeze = [".*"]

        if not isinstance(layers_to_unfreeze, (list, tuple, np.ndarray)):
            layers_to_unfreeze = [layers_to_unfreeze]

        for pattern in layers_to_unfreeze:
            selected_parameters = select_parameters(self._model, pattern)

            # Check that the pattern matches at least one parameter
            # This is a safeguard to avoid wrong patterns
            if len(selected_parameters) == 0:
                msg_error = f'No parameters were found with the pattern {pattern}.'
                raise ValueError(msg_error)

            for param in selected_parameters:
                param.requires_grad = True

    def tokenize(self, sequences):
        """
        Tokenizes protein sequences.

        Parameters
        ----------
        sequences : list
            List of protein sequences to tokenize in FASTA format.

        Returns
        -------
        tokens : torch.Tensor of shape (n_sequences, n_tokens)
            Tokenized sequences.
        attention_mask : torch.Tensor of shape (n_sequences, n_tokens)
            Attention mask for the tokenized molecules. Returns None if the model does not use attention masks.

        Raises
        ------
        AssertionError
            If the sequences are not in FASTA format.
        AssertionError
            If the sequences have different lengths.

        """
        attention_mask = None
        sequence_formats = np.unique(utils.guess_input_formats(sequences))

        msg_error = f'Only FASTA format is supported. Got {sequence_formats}.'
        assert sequence_formats[0] == 'FASTA' and sequence_formats.size == 1, msg_error
        assert np.unique([len(s) for s in sequences]).size == 1, f'All sequences must have the same length.'

        if not isinstance(sequences, (list, tuple, np.ndarray, torch.Tensor)):
            sequences = [sequences]

        if self._padding_length is None:
            padding = 'longest'
            max_length = None
        else:
            padding = 'max_length'
            max_length = self._padding_length

        if self._model_type == 'esm':
            tokens = []

            for seq in sequences:
                _, _, token = self._tokenizer([('sequence', seq)])
                tokens.append(token)

            tokens = torch.cat(tokens)
        else:
            if isinstance(sequences, np.ndarray):
                sequences = sequences.tolist()

            # Need to add spaces between amino acids for some models (e.g. T5Tokenizer)
            sequences = [' '.join(seq) for seq in sequences]

            # Truncation is False because we want to keep the full sequence. It will fail 
            # if the sequence is longer than the maximum length, so we avoid bad surprises.
            output = self._tokenizer(sequences, add_special_tokens=True, return_tensors='pt', padding=padding, truncation=False, max_length=max_length)

            tokens = output['input_ids']
            if 'attention_mask' in output:
                attention_mask = output['attention_mask']

        # Move tensors to device
        tokens = tokens.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        return tokens, attention_mask

    def embed(self, tokenized_sequences, attention_mask=None, return_probabilities=False):
        """
        Computes embedding vectors for protein sequences.

        Parameters
        ----------
        tokenized_sequences : torch.Tensor of shape (n_sequences, n_tokens) or dict
            List of tokenized protein sequences to embed.
        return_probabilities : bool
            Whether to return the probabilities of amino acids at each position per sequence.

        Returns
        -------
        embeddings : torch.Tensor of shape (n_sequences, n_features)
            Embedding vectors for each sequence. The number of features depends on the requested 
            embedding type. With the 'avg' embedding type, the number of features is equal to the 
            out_features size of the model, while for the 'residue' embedding type, the number 
            of features will be equal to the out_features size times the number of tokens in the 
            sequence.
        probabilities : torch.Tensor of shape (n_sequences, n_tokens, n_amino_acids)
            Probabilities of amino acids at each position per sequence. If return_probabilities is True.

        """
        if self._model_type == 'esm':
            results = self._model(tokenized_sequences, repr_layers=[33])
            embeddings = results['representations'][33]
        else:
            results = self._model(input_ids=tokenized_sequences, attention_mask=attention_mask)
            embeddings = results.last_hidden_state

        # Either flatten the embeddings or average it
        if self._embedding_type == 'residue':
            features = embeddings.reshape(-1)
        else:
            features = torch.mean(embeddings, 1)

        if return_probabilities and self._vocabulary_mask is not None:
            if 'logits' not in results:
                msg_error = f'Cannot return probabilities with model {self._model_name}. Please set return_probabilities to False.'
                raise ValueError(msg_error)

            # We keep only the probabilities of the standard amino acids, not the special tokens
            logits = results['logits'][:, 1:len(tokenized_sequences[0]) + 1, self._vocabulary_mask]
            # We apply softmax to get probabilities from logits
            softmax = torch.nn.Softmax(dim=-1)
            probabilities = torch.tensor([softmax(l) for l in logits])

            return features, probabilities

        return features

    def transform(self, sequences, return_probabilities=False):
        """
        Computes embedding vectors for protein sequences.

        Parameters
        ----------
        sequences : list
            List of protein sequences to embed in FASTA format.
        return_probabilities : bool
            Whether to return the probabilities of amino acids at each position per sequence.

        Returns
        -------
        embeddings : ndarray of shape (n_sequences, n_features)
            Embedding vectors for each sequence.
        probabilities : ndarray of shape (n_sequences, n_tokens, n_amino_acids)
            Probabilities of amino acids at each position per sequence. If return_probabilities is True.

        """
        tokenized_sequences, attention_mask = self.tokenize(sequences)
        results = self.embed(tokenized_sequences, attention_mask, return_probabilities)

        return results

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
