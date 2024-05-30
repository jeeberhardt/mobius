#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Graph
#

import graphein.molecule as gm
import networkx as nx
import numpy as np
import torch
from grakel.utils import graph_from_networkx
from rdkit import Chem
from torch_geometric.data import Batch, Data

from .utils import MolFromHELM


class Graph:
    """
    A class for the graph representation of molecules.

    """

    def __init__(self, input_type='helm', output_type='graph', config=None, node_labels=None, edge_labels=None, HELM_parser='mobius', HELM_extra_library_filename=None):
        """
        Constructs a new instance of the Graph class.

        Parameters
        ----------
        input_type : str, default 'helm_rdkit'
            Input format of the polymers. The options are 'fasta', 
            'helm_rdkit', 'helm', or 'smiles'.
        output_type : str, default 'graph'
            Output graph format. The options are 'graph' or 'pyg'.
            Use `graph` when using `GPGModel` and `pyg` when using `GPGNNModel`.
        config : graphein.molecule.MoleculeGraphConfig, optional (default=None)
            The configuration of the molecule graph.
        node_labels : str, optional (default=None)
            The node labels to be used. The options are 'element','degree', 
            'valence', 'hybridization', 'aromaticity', 'formal_charge',
            'total_num_h', etc, ... (see the graphein documentation for more 
            options).
        edge_labels : str, optional (default=None)
            The edge labels to be used. The options are 'bond_type', 
            'conjugated', 'stereo', 'in_ring', 'ring_size', etc, ... 
            (see the graphein documentation for more options).
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
        msg_error = f'Format {input_type.lower()} not handled. Please use FASTA, HELM or SMILES format.'
        assert input_type.lower() in ['fasta', 'helm', 'smiles'], msg_error

        msg_error = f'Output type {output_type.lower()} not handled. Please use graph or pyg.'
        assert output_type.lower() in ['graph', 'pyg'], msg_error

        self._input_type = input_type.lower()
        self._output_type = output_type.lower()
        self._HELM_parser = HELM_parser.lower()
        self._HELM_extra_library_filename = HELM_extra_library_filename

        if config is not None:
            self._config = config
        else:
            self._config = gm.MoleculeGraphConfig()

        if not isinstance(node_labels, (list, tuple, np.ndarray)) and node_labels is not None:
            self._node_labels = [node_labels]
        else:
            self._node_labels = node_labels

        if not isinstance(edge_labels, (list, tuple, np.ndarray)) and edge_labels is not None:
            self._edge_labels = [edge_labels]
        else:
            self._edge_labels = edge_labels

        if self._output_type == 'graph' and isinstance(node_labels, (list, tuple, np.ndarray)):
            msg_error = f'Only one node label ({self._node_labels}) is allowed when using the graph output type.'
            assert len(self._node_labels) == 1, msg_error
            self._node_labels = self._node_labels[0]

        if self._output_type == 'graph' and isinstance(edge_labels, (list, tuple, np.ndarray)):
            msg_error = f'Only one edge label ({self._edge_labels}) is allowed when using the graph output type.'
            assert len(self._edge_labels) == 1, msg_error
            self._edge_labels = self._edge_labels[0]

        self._graph_labels = None

    def _convert_nx_to_pyg(self, G):
        edge_feature_types = {}

        # Initialise dict used to construct Data object & Assign node ids as a feature
        data = {"node_id": list(G.nodes())}

        G = nx.convert_node_labels_to_integers(G)

        # Construct Edge Index
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        if self._node_labels is not None:
            data['node_attr'] = []

            # Add node features
            for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
                tmp = []

                for key, value in feat_dict.items():
                    if str(key) in self._node_labels:
                        tmp = np.concatenate((tmp,  np.atleast_1d(value)))

                data['node_attr'].append(tmp)

            data['node_attr'] = torch.from_numpy(np.asarray(data['node_attr'])).float()

        if self._edge_labels is not None:
            data['edge_attr'] = []

            # Add edge features
            for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
                tmp = []

                for key, value in feat_dict.items():
                    if str(key) in self._edge_labels:
                        edge_feature_types.setdefault(key, [])

                        try:
                            i = edge_feature_types[key].index(value)
                        except ValueError:
                            edge_feature_types[key].append(value)
                            i = len(edge_feature_types[key]) - 1

                        tmp.append(i)

                data['edge_attr'].append(tmp)

            data['edge_attr'] = torch.from_numpy(np.asarray(data['edge_attr'])).float()

        if self._graph_labels is not None:
            # Add graph-level features
            for feat_name in G.graph:
                if str(feat_name) in self._node_labels:
                    data[str(feat_name)] = [G.graph[feat_name]]

        data["edge_index"] = edge_index.view(2, -1)

        data = Data.from_dict(data)
        data.num_nodes = G.number_of_nodes()

        return data

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

        if self._output_type == 'graph':
            graphs = [gm.construct_graph(smiles=s, config=self._config) for s in smiles]
            graphs = np.array(list(graph_from_networkx(graphs, node_labels_tag=self._node_labels,
                                                       edge_labels_tag=self._edge_labels)))

            """
            # Still not sure I need that part
            edge_types = {}

            if self._edge_labels_tag is not None:
                i = 0

                for graph in graphs:
                    edges = graph[2]

                    for key, value in edges.items():
                        if value not in edge_types:
                            edge_types[value] = i
                            i += 1
                        edges[key] = edge_types[value]
            """
        else:
            graphs = [self._convert_nx_to_pyg(gm.construct_graph(smiles=s, config=self._config)) for s in smiles]
            graphs = Batch.from_data_list(graphs)

        return graphs

