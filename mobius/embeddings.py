#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Protein Embeddings
#

import importlib
import re

import esm
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import utils


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        """
        Initializes the LoRALayer class.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        rank : int, default : 8
            Rank of the LoRA layer.
        alpha : int, default : 16
            Scaling factor of the LoRA layer.
            
        """
        super().__init__()
        self.scaling = alpha / rank
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        """
        Computes the output of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.

        """
        x = self.scaling * (x @ self.A @ self.B)
        return x
    
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank=4, alpha=16):
        """
        Initializes the LinearWithLoRA class.

        Parameters
        ----------
        linear : torch.nn.Linear
            Linear layer.
        rank : int, default : 4
            Rank of the LoRA layer.
        alpha : int, default : 16
            Scaling factor of the LoRA layer.

        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        """
        Computes the output of the layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.

        """
        return self.linear(x) + self.lora(x)
    

def replace_layers_with_lora(model, patterns, rank=4, alpha=16):
    """
    Replaces linear layers with LoRA layers.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    patterns : str or List of str
        Patterns to select the layers to replace.
    rank : int, default : 4
        Rank of the LoRA layer.
    alpha : int, default : 16
        Scaling factor of the LoRA layer.

    Raises
    ------
    AssertionError
        If the layer to replace is not a torch.nn.Linear.

    """
    for m_name, module in dict(model.named_modules()).items():
        for c_name, layer in dict(module.named_children()).items():
            if any(re.search(pattern, f'{m_name}.{c_name}') for pattern in patterns):
                msg_error = f'LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}.'
                assert isinstance(layer, torch.nn.Linear), msg_error
                setattr(module, c_name, LinearWithLoRA(layer, rank, alpha))


def select_parameters(model, parameters_names):
    """
    Selects parameters from a model based on their names.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    parameter_names : str or List of str (default=None)
        Names of the parameters to select.

    Returns
    -------
    selected_parameters : List of torch.nn.Parameter
        Selected parameters.

    """
    selected_parameters = []

    if not isinstance(parameters_names, (list, tuple, np.ndarray)):
        parameters_names = np.array([parameters_names])

    for name, param in model.named_parameters():
        if any(re.search(pattern, name) for pattern in parameters_names):
            selected_parameters.append(param)

    return selected_parameters


class ProteinEmbedding:

    def __init__(self, pretrained_model_name='esm1b_t33_650M_UR50S', embedding_type='avg', 
                 device=None, model_name=None, tokenizer_name=None,
                 layers_to_finetune=None, lora=False, lora_rank=4, lora_alpha=16):
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
            LoRA layers. If False, apply traditional fine-tuning strategy (modify weights directly).
        lora_rank : int, default : 4
            Rank of the LoRA layer. Only if lora=True.
        lora_alpha : int, default : 16
            Scaling factor of the LoRA layer. Only if lora=True.

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

        if self._layers_to_finetune is not None:
            # Freeze all parameters
            for param in self._model.parameters():
                param.requires_grad = False

            # Either replace layers with LoRA or apply traditional fine-tuning strategy
            if self._lora:
                replace_layers_with_lora(self._model, self._layers_to_finetune, self._lora_rank, self._lora_alpha)
            else:
                layers_to_finetune = select_parameters(self._model, self._layers_to_finetune)

                for param in layers_to_finetune:
                    param.requires_grad = True

    def tokenize(self, sequences):
        """
        Tokenizes protein sequences.

        Parameters
        ----------
        sequences : list
            List of protein sequences to tokenize.

        Returns
        -------
        tokens : torch.Tensor of shape (n_sequences, n_tokens)
            Tokenized sequences.

        Raises
        ------
        AssertionError
            If the sequences are not in FASTA format.
        AssertionError
            If the sequences have different lengths.

        """
        sequence_formats = np.unique(utils.guess_input_formats(sequences))

        msg_error = f'Only FASTA format is supported. Got {sequence_formats}.'
        assert sequence_formats[0] == 'FASTA' and sequence_formats.size == 1, msg_error
        assert np.unique([len(s) for s in sequences]).size == 1, f'All sequences must have the same length.'

        if not isinstance(sequences, (list, tuple, np.ndarray, torch.Tensor)):
            sequences = [sequences]

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

            tokens = self._tokenizer(sequences, add_special_tokens=True, return_tensors='pt', padding="longest")['input_ids']

        # Move tensors to device
        tokens = tokens.to(self._device)

        return tokens

    def embed(self, tokenized_sequences, return_probabilities=False):
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
            results = self._model(input_ids=tokenized_sequences)
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
            List of protein sequences to embed.
        return_probabilities : bool
            Whether to return the probabilities of amino acids at each position per sequence.

        Returns
        -------
        embeddings : ndarray of shape (n_sequences, n_features)
            Embedding vectors for each sequence.
        probabilities : ndarray of shape (n_sequences, n_tokens, n_amino_acids)
            Probabilities of amino acids at each position per sequence. If return_probabilities is True.

        """
        tokenized_sequences = self.tokenize(sequences)
        results = self.embed(tokenized_sequences, return_probabilities)

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
