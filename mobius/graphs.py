#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Graph
#

import graphein.molecule as gm
import numpy as np
from grakel.utils import graph_from_networkx
from rdkit import Chem

from .utils import MolFromHELM


class Graph:

    """
    A class for the graph representation of molecules.

    """

    def __init__(self, input_type='helm', config=None, node_labels=None, edge_labels=None, HELM_parser='mobius', HELM_extra_library_filename=None):
        """
        Constructs a new instance of the Graph class.

        Parameters
        ----------
        input_type : str, default 'helm_rdkit'
            Input format of the polymers. The options are 'fasta', 
            'helm_rdkit', 'helm', or 'smiles'.
        config : graphein.molecule.MoleculeGraphConfig, optional (default=None)
            The configuration of the molecule graph.
        node_labels : str, optional (default=None)
            The node labels to be used. The options are 'atom', 'element',
            'degree', 'valence', 'hybridization', 'aromaticity', 'formal_charge',
            'num_H', 'isotope', etc, ... (see the RDKit documentation for more options).
        edge_labels : str, optional (default=None)
            The edge labels to be used. The options are 'bond_type', 'conjugated',
            'stereo', 'in_ring', 'ring_size', 'ring_membership', 'num_aromatic_bonds',
            etc, ... (see the RDKit documentation for more options).
        HELM_parser : str, optional (default='mobius')
            The HELM parser to be used. It can be 'mobius' or 'rdkit'. 
            When using 'rdkit' parser, only D or L standard amino acids
            are supported with disulfide bridges for macrocyclic polymers.
            Using the (slower) internal 'mobius' HELM parser, all kind of
            scaffolds and monomers can be supported. Monomers library can 
            be easily extended by providing an extra HELM Library file using 
            the `HELM_extra_library_filename` parameter.
        HELM_extra_library_filename : str, optional (default=None)
            The path to a HELM Library file containing extra monomers.
            Extra monomers will be added to the internal monomers library. 
            Internal monomers can be overriden by providing a monomer with
            the same MonomerID.

        Notes
        -----
        When using HELM format as input, the fingerprint of a same molecule 
        can differ depending on the HELM parser used. The fingerprint of a 
        molecule in FASTA format is guaranteed to be the same if the same 
        molecule in HELM format uses `rdkit` for the parsing, but won't be 
        necessarily with the internal HELM parser.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm', 'smiles'], msg_error

        self._input_type = input_type.lower()
        self._HELM_parser = HELM_parser.lower()
        self._HELM_extra_library_filename = HELM_extra_library_filename
        if config is not None:
            self._config = config
        else:
            self._config = gm.MoleculeGraphConfig()
        self._node_labels_tag = node_labels
        self._edge_labels_tag = edge_labels
        self._edge_types = {}

    def transform(self, polymers):
        """
        Transforms the input polymers into graphs.

        Parameters
        ----------
        polymers : str, list or numpy.ndarray
            A list of polymers or a single polymer to be transformed.

        Returns
        -------
        graphs : numpy.ndarray
            A numpy array of graphs.

        """
        if not isinstance(polymers, (list, tuple, np.ndarray)):
            polymers = [polymers]

        try:
            if self._input_type == 'fasta':
                smiles = [Chem.MolToSmiles(Chem.rdmolfiles.MolFromFASTA(c)) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'rdkit':
                smiles = [Chem.MolToSmiles(Chem.rdmolfiles.MolFromHELM(c)) for c in polymers]
            elif self._input_type == 'helm' and self._HELM_parser == 'mobius':
                smiles = [Chem.MolToSmiles(c) for c in MolFromHELM(polymers, self._HELM_extra_library_filename)]
            else:
                smiles = polymers
        except AttributeError:
            print('Error: there are issues with the input molecules')
            print(polymers)

        graphs = [gm.construct_graph(smiles=s, config=self._config) for s in smiles]
        graphs = np.array(list(graph_from_networkx(graphs, node_labels_tag=self._node_labels_tag, edge_labels_tag=self._edge_labels_tag)))

        #"""
        if self._edge_labels_tag is not None:
            i = 0

            for graph in graphs:
                edges = graph[2]

                for key, value in edges.items():
                    if value not in self._edge_types:
                        self._edge_types[value] = i
                        i += 1
                    edges[key] = self._edge_types[value]
        #"""

        return graphs
