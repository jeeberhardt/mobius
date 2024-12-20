#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Protein Embeddings
#

import importlib
import inspect

import esm
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .utils import replace_layers_with_lora, select_parameters
from .. import utils


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    
    Taken from https://github.com/facebookresearch/esm/blob/main/esm/data.py#L253 and
    modified because their version is not convenient.

    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch, padding='longest', truncation=False, max_length=None):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)

        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        size_seq_encoded_list = np.asarray([len(seq_encoded) for seq_encoded in seq_encoded_list])

        if padding == 'longest':
            max_length = np.max(size_seq_encoded_list)
        elif padding == 'max_length':
            assert max_length is not None, 'When padding is set to `max_length`, the `max_length` argument must be defined.'

        if truncation:
            for i in np.argwhere(size_seq_encoded_list > max_length).flatten():
                seq_encoded_list[i] = seq_encoded_list[i][:max_length]
        else:
            msg_error = f'Some sequences are longer than `max_length` ({max_length}), but truncation is not allowed. '
            msg_error += 'Please increase `max_length` or set truncation to True.'
            assert np.argwhere(size_seq_encoded_list > max_length).size == 0, msg_error
        
        tokens = torch.empty(
            (
                batch_size,
                max_length + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens


class ProteinEmbedding:

    def __init__(self, pretrained_model_name='esm1b_t33_650M_UR50S', embedding_type='avg', 
                 model_name=None, tokenizer_name=None,
                 layers_to_finetune=None, lora=False, lora_rank=4, lora_alpha=8,
                 padding_length=None, add_extra_space=True, device=None):
        """
        Initializes the ProteinEmbedding class.

        Parameters
        ----------
        pretrained_model_name : str
            Name of the encoder model. The model will be downloaded from the huggingface repository, 
            except for ESM models. Suggestions for pretrained_model_name are:
            - ESM models: esm1b_t33_650M_UR50S, esm1v_t33_650M_UR90S_1, esm2_t36_3B_UR50D
            - ProtT5: Rostlab/prot_t5_xl_uniref50, Rostlab/prot_bert
            - Others: TianlaiChen/PepMLM-650M
        embedding_type : str
            Type of embedding to use, either 'residue' or 'avg' (average). The number of output features 
            depends on the requested embedding type. With the 'avg' embedding type, the number of features 
            is equal to the out_features size of the model, while for the 'residue' embedding type, the 
            number of features will be equal to the out_features size times the number of tokens in the 
            sequence.
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
        add_extra_space : bool, default : True
            During tokenization, add an extra space between amino acid characters. This is required
            for some models, like T5Tokenizer. If False, the sequences will be tokenized without extra 
            spaces.
        device : str or torch.device, default : None
            Device on which to run the model. Per default, the device is set to 
            'cuda' if available, otherwise to 'cpu'.

        Notes
        -----
        - Prot_t5_xl_uniref50: Use the `T5EncoderModel` (embeddings only) or `T5ForConditionalGeneration` 
          (embedding and logits) model and `T5Tokenizer` tokenizer.

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
        self._add_extra_space = add_extra_space

        # https://github.com/google/sentencepiece?tab=readme-ov-file#whitespace-is-treated-as-a-basic-symbol
        meta_symbol = u"\u2581"

        if 'esm' in pretrained_model_name:
            self._model_type = 'esm'
            self._model, alphabet = esm.pretrained.load_model_and_alphabet(pretrained_model_name)
            self._tokenizer = BatchConverter(alphabet)
            # Get BOS, EOS and PAD tokens
            self._bos_token = self._tokenizer.alphabet.cls_idx
            self._eos_token = self._tokenizer.alphabet.eos_idx
            self._padding_token = self._tokenizer.alphabet.padding_idx
            self._vocab = alphabet.all_toks
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

            # Get BOS, EOS and PAD tokens
            self._bos_token = self._tokenizer.bos_token_id
            self._eos_token = self._tokenizer.eos_token_id
            self._padding_token = self._tokenizer.pad_token_id
            # Remove the meta symbol from tokens used in sentencepiece
            self._vocab = [token.replace(meta_symbol, "") for token in self._tokenizer.get_vocab()]

        # Get idx of the natural amino acids in the pLM vocab, so we select only these
        # And we keep the same amino acid order for each pLM
        self._vocabulary_idx = np.array([self._vocab.index(aa) for aa in self._standard_amino_acids if aa in self._vocab])

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

        if not isinstance(sequences, (list, tuple, np.ndarray, torch.Tensor)):
            sequences = [sequences]

        if self._padding_length is None:
            padding = 'longest'
            max_length = None
        else:
            padding = 'max_length'
            max_length = self._padding_length

        if self._model_type == 'esm':
            _, _, tokens = self._tokenizer([('sequence', seq) for seq in sequences], padding=padding, truncation=False, max_length=max_length)
        else:
            if isinstance(sequences, np.ndarray):
                sequences = sequences.tolist()

            # Need to add spaces between string characters (amino acids) for some models (e.g. T5Tokenizer)
            if self._add_extra_space:
                sequences = [' '.join(seq) for seq in sequences]

            # Truncation is False because we want to keep the full sequence. It will fail 
            # if the sequence is longer than the maximum length, so we avoid bad surprises.
            output = self._tokenizer(sequences, add_special_tokens=True, return_tensors='pt', padding=padding, truncation=False, max_length=max_length)
            tokens = output['input_ids']

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
        embeddings : torch.Tensor of shape (n_sequence, n_residues, n_features) or list of torch.Tensors of shape (n_residues, n_features)
            Embedding vectors for each sequence. The number of features depends on the requested 
            embedding type. With the 'avg' embedding type, the number of features is equal to the 
            out_features size of the model, while for the 'residue' embedding type, the number 
            of features will be equal to the out_features size times the number of tokens in the 
            sequence. Returns a list of torch.Tensors if the sequences have different lengths.
        probabilities : torch.Tensor of shape (n_sequences, n_residues, n_amino_acids) or list of torch.Tensors of shape (n_residues, n_amino_acids)
            Probabilities of amino acids at each position per sequence. If return_probabilities is 
            True. Returns a list of torch.Tensors if the sequences have different lengths. The 
            probabilities for each residue follows this order: 'A', 'C', 'D', 'E', 'F', 'G', 'H', 
            'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'.

        """
        extra_arguments = {}

        # Mask for selecting sequence and not the BOS, EOS and PAD tokens
        sequence_mask = (tokenized_sequences != self._padding_token) \
                      & (tokenized_sequences != self._eos_token) \
                      & (tokenized_sequences != self._bos_token)
        # Check if the sequence have all the same length, compare first sequence to the rest
        are_sequences_all_same_length = bool((sequence_mask == sequence_mask[0]).all())

        if self._model_type == 'esm':
            results = self._model(tokenized_sequences, repr_layers=[33])
            embeddings = results['representations'][33]
        else:
            # Make it compatible with T5EncoderModel and T5ForConditionalGeneration models
            model_arguments = inspect.getargspec(self._model)[0]

            if 'decoder_input_ids' in model_arguments:
                extra_arguments['decoder_input_ids'] = tokenized_sequences

            results = self._model(input_ids=tokenized_sequences, **extra_arguments)

            if 'last_hidden_state' in results:
                embeddings = results.last_hidden_state
            elif 'encoder_last_hidden_state' in results:
                embeddings = results.encoder_last_hidden_state
            else:
                raise RuntimeError(f'Cannot find last hidden state (embedding) from model\'s output ({results.keys()}).')

        # Either flatten the embeddings or average it
        if self._embedding_type == 'residue':
            if are_sequences_all_same_length:
                # if all the same length, we can use some array-tricks
                selected_elements = embeddings[sequence_mask.unsqueeze(-1).expand_as(embeddings)]
                num_trues = sequence_mask.sum(dim=1).max().item()  # This assumes all batches have the same number of Trues
                embeddings = selected_elements.reshape(embeddings.shape[0], num_trues, embeddings.shape[2])
                # Flatten the embeddings, so one feature vector for each sequence
                new_shape = (embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
                features = embeddings.reshape(new_shape)
            else:
                # If not all the same length, then we need to iterate through all..
                features = [e[m] for e, m in zip(embeddings, sequence_mask)]
        else:
            # If we use average embeddings, no need to worry about sequence lengths
            # Source: https://stackoverflow.com/a/69314298
            denom = torch.sum(sequence_mask, -1, keepdim=True)
            features = torch.sum(embeddings * sequence_mask.unsqueeze(-1), dim=1) / denom

        if return_probabilities:
            if 'logits' not in results:
                msg_error = f'Cannot return probabilities with model {self._model_name}. Please set return_probabilities to False.'
                raise ValueError(msg_error)

            logits = results['logits']

            if are_sequences_all_same_length:
                selected_elements = logits[sequence_mask.unsqueeze(-1).expand_as(logits)]
                num_trues = sequence_mask.sum(dim=1).max().item()  # This assumes all batches have the same number of Trues
                logits = selected_elements.reshape(logits.shape[0], num_trues, logits.shape[2])
                # We keep only the probabilities of the standard amino acids, not the special tokens
                logits = logits[:, :, self._vocabulary_idx]
            else:
                # If not all the same length, then we need to iterate through all..
                # We keep only the probabilities of the standard amino acids, not the special tokens
                logits = [l[m][:, self._vocabulary_idx] for l, m in zip(logits, sequence_mask)]

            # We apply softmax to get probabilities from logits
            softmax = torch.nn.Softmax(dim=-1)
            probabilities = [softmax(l) for l in logits]

            try:
                probabilities = torch.stack(probabilities)
            except:
                pass

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
