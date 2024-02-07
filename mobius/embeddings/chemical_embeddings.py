#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Chemical Embeddings
#

import importlib

import numpy as np
import torch
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer

from .utils import replace_layers_with_lora, select_parameters
from .. import utils


class ChemicalEmbedding:

    def __init__(self, pretrained_model_name, input_type='helm', embedding_type='avg', 
                 device=None, model_name=None, tokenizer_name=None,
                 layers_to_finetune=None, lora=False, lora_rank=4, lora_alpha=8,
                 padding_length=None, add_extra_space=False):
        """
        Initializes the ProteinEmbedding class.

        Parameters
        ----------
        pretrained_model_name : str
            Name of the encoder model. The model will be downloaded from the huggingface repository.
            Suggestions for pretrained_model_name are:
            - DeepChem's ChemBERTa
        input_type: str, default : 'HELM'
            Format of the input molecules, either 'HELM' or 'SMILES'.
        embedding_type : str
            Type of embedding to use, either 'atom' or 'avg' (average). The number of output features 
            depends on the requested embedding type. With the 'avg' embedding type, the number of features 
            is equal to the out_features size of the model, while for the 'atom' embedding type, the 
            number of features will be equal to the out_features size times the number of tokens in the 
            molecule.
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
        add_extra_space : bool, default : False
            During tokenization, add an extra space between smiles characters.

        Raises
        ------
        AssertionError
            If the embedding type is not supported.

        AssertionError
            If the input type is not supported.

        """
        assert input_type in ['fasta', 'helm', 'smiles'], 'Only FASTA, HELM and SMILES format are supported.'
        assert embedding_type in ['atom', 'avg'], 'Only average (avg) and residue embeddings are supported.'

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = device
        self._pretrained_model_name = pretrained_model_name
        self._input_type = input_type.lower()
        self._embedding_type = embedding_type
        self._layers_to_finetune = layers_to_finetune
        self._lora = lora
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._padding_length = padding_length
        self._add_extra_space = add_extra_space

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

        self._padding_token = self._tokenizer.pad_token_id

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

    def tokenize(self, molecules):
        """
        Tokenizes molecules.

        Parameters
        ----------
        molecules : list
            List of molecules to tokenize in HELM or SMILES format.

        Returns
        -------
        output : dict
            Dictionary containing the following fields:
            - tokens : torch.Tensor of shape (n_molecules, n_tokens)
                Tokenized molecules.
            - attention_mask : torch.Tensor of shape (n_molecules, n_tokens) or None
                Attention mask for the tokenized molecules. None if the model 
                does not use attention masks.

        """
        attention_mask = torch.Tensor([])

        if not isinstance(molecules, (list, tuple, np.ndarray, torch.Tensor)):
            molecules = [molecules]

        if self._input_type == 'helm':
            molecules = utils.MolFromHELM(molecules)
            molecules = [Chem.MolToSmiles(mol) for mol in molecules]
        elif self._input_type == 'fasta':
            molecules = [Chem.MolToSmiles(Chem.MolFromFasta(mol)) for mol in molecules]

        if self._padding_length is None:
            padding = 'longest'
            max_length = None
        else:
            padding = 'max_length'
            max_length = self._padding_length

        # Need to add spaces between string characters for some models (e.g. T5Tokenizer)
        if self._add_extra_space:
            molecules = [' '.join(seq) for seq in molecules]

        # Truncation is False because we want to keep the full sequence. It will fail 
        # if the sequence is longer than the maximum length, so we avoid bad surprises.
        output = self._tokenizer(molecules, add_special_tokens=True, return_tensors='pt', padding=padding, truncation=False, max_length=max_length)

        tokens = output['input_ids']
        if 'attention_mask' in output:
            attention_mask = output['attention_mask']

        # Move tensors to device
        tokens = tokens.to(self._device)
        attention_mask = attention_mask.to(self._device)
        
        output = {'tokens': tokens, 'attention_mask': attention_mask}

        return output

    def embed(self, tokenized_molecules, attention_mask=None):
        """
        Computes embedding vectors for molecules.

        Parameters
        ----------
        tokenized_molecules : torch.Tensor of shape (n_molecules, n_tokens) or dict
            List of tokenized molecules to embed.
        attention_mask : torch.Tensor of shape (n_molecules, n_tokens), default : None
            Attention mask for the tokenized molecules.

        Returns
        -------
        embeddings : torch.Tensor of shape (n_molecules, n_features)
            Embedding vectors for each molecule. The number of features depends on the requested 
            embedding type. With the 'avg' embedding type, the number of features is equal to the 
            out_features size of the model, while for the 'atom' embedding type, the number 
            of features will be equal to the out_features size times the number of tokens in the 
            molecule.

        """
        results = self._model(input_ids=tokenized_molecules, attention_mask=attention_mask)
        embeddings = results.last_hidden_state

        # Either flatten the embeddings or average it
        if self._embedding_type == 'atom':
            new_shape = (embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
            features = embeddings.reshape(new_shape)
        else:
            # Source: https://stackoverflow.com/a/69314298
            mask = tokenized_molecules != self._padding_token
            denom = torch.sum(mask, -1, keepdim=True)
            features = torch.sum(embeddings * mask.unsqueeze(-1), dim=1) / denom

        return features

    def transform(self, molecules):
        """
        Computes embedding vectors for molecules.

        Parameters
        ----------
        molecules : list
            List of molecules to embed in HELM or SMILES format.

        Returns
        -------
        embeddings : ndarray of shape (n_molecules, n_features)
            Embedding vectors for each molecule.

        """
        tokenized_molecules = self.tokenize(molecules)
        results = self.embed(tokenized_molecules['tokens'], tokenized_molecules['attention_mask'])

        return results
