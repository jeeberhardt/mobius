#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Graph
#

import networkx as nx
import numpy as np
import torch
from grakel.utils import graph_from_networkx
from rdkit import Chem
from torch_geometric.data import Batch, Data
from rdkit.Chem.rdchem import HybridizationType, BondStereo

from .utils import MolFromHELM


def convert_element_to_one_hot(atom):
    common_elements = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I']

    element = atom.GetSymbol()
    one_hot = np.zeros(shape=len(common_elements) + 1)

    try:
        one_hot[common_elements.index(element)] = 1
    except:
        one_hot[-1] = 1

    return one_hot


def convert_degree_to_one_hot(atom, max_degree=6):
    degree = atom.GetDegree()
    one_hot = np.zeros(shape=max_degree)
    one_hot[min(degree, max_degree)] = 1

    return one_hot


def convert_hydrogens_to_one_hot(atom, max_hydrogens=5):
    num_hydrogens = atom.GetTotalNumHs()
    one_hot = np.zeros(shape=max_hydrogens)
    one_hot[min(num_hydrogens, max_hydrogens)] = 1

    return one_hot


def convert_hybridization_to_one_hot(atom):
    hybridization_types = [HybridizationType.S, 
                           HybridizationType.SP, 
                           HybridizationType.SP2, 
                           HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2,
                           HybridizationType.OTHER, HybridizationType.UNSPECIFIED]
    
    hybridization = atom.GetHybridization()
    one_hot = np.zeros(shape=len(hybridization_types))
    one_hot[hybridization_types.index(hybridization)] = 1

    return one_hot


def convert_formal_charge_to_integer(atom):
    return np.asarray([atom.GetFormalCharge()])


def convert_radical_electrons_to_integer(atom):
    return np.asarray([atom.GetNumRadicalElectrons()])


def convert_aromatic_to_bit(atom):
    return np.asarray([int(atom.GetIsAromatic())])


def convert_chirality_to_one_hot(atom):
    chirality_types = ["", "R", "S"]
    
    if atom.HasProp('_CIPCode'):
        chirality_type = atom.GetProp('_CIPCode')
    else:
        chirality_type = ""
        
    one_hot = np.zeros(shape=len(chirality_types))
    one_hot[chirality_types.index(chirality_type)] = 1

    return one_hot


def convert_bond_type_to_string(bond):
    bond_types = {Chem.rdchem.BondType.SINGLE: 'single', 
                  Chem.rdchem.BondType.DOUBLE: 'double',
                  Chem.rdchem.BondType.TRIPLE: 'triple', 
                  Chem.rdchem.BondType.AROMATIC: 'aromatic'}
    
    try:
        return bond_types[bond.GetBondType()]
    except:
        return 'other'


def convert_bond_type_to_one_hot(bond):
    bond_types = [Chem.rdchem.BondType.SINGLE, 
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, 
                  Chem.rdchem.BondType.AROMATIC]
    
    bond_type = bond.GetBondType()
    one_hot = np.zeros(shape=len(bond_types))
    one_hot[bond_types.index(bond_type)] = 1

    return one_hot


def convert_conjugation_to_bit(bond):
    return np.asarray([int(bond.GetIsConjugated())])


def convert_ring_to_bit(bond):
    return np.asarray([int(bond.IsInRing())])


def convert_stereo_to_one_hot(bond):
    stereo_types = [BondStereo.STEREONONE, 
                    BondStereo.STEREOANY, 
                    BondStereo.STEREOZ, 
                    BondStereo.STEREOE]
    
    stereo = bond.GetStereo()
    one_hot = np.zeros(shape=len(stereo_types))
    one_hot[stereo_types.index(stereo)] = 1

    return one_hot


class Graph:
    """
    A class for the graph representation of molecules.

    """

    def __init__(self, input_type='helm', output_type='graph', HELM_parser='mobius', HELM_extra_library_filename=None):
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

        self._node_labels = 'node_attr'
        self._edge_labels = 'edge_attr'
        self._graph_labels = None

    def _convert_nx_to_pyg(self, G):
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
                        tmp = np.concatenate((tmp,  np.atleast_1d(value)))

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

    def _construct_features_graph(self, mol):
        # Initialize an empty graph
        G = nx.Graph()
    
        node_featurizers = [convert_element_to_one_hot,
                            convert_degree_to_one_hot,
                            convert_formal_charge_to_integer, 
                            convert_radical_electrons_to_integer,
                            convert_hybridization_to_one_hot,
                            convert_aromatic_to_bit,
                            convert_hydrogens_to_one_hot,
                            convert_chirality_to_one_hot]
    
        # Add nodes with atom indices as node IDs
        for atom in mol.GetAtoms():
            node_id = f'{atom.GetSymbol()}:{atom.GetIdx()}'
    
            node_features = np.array([])
            for node_featurizer in node_featurizers:
                node_features = np.concatenate([node_features, node_featurizer(atom)])
            
            G.add_node(node_id, node_attr=node_features)

        edge_featurizers = [convert_bond_type_to_one_hot,
                            convert_conjugation_to_bit,
                            convert_ring_to_bit,
                            convert_stereo_to_one_hot]
    
        # Add edges with bond information
        for bond in mol.GetBonds():
            b_atom = bond.GetBeginAtom()
            e_atom = bond.GetEndAtom()
            edge_id = (f'{b_atom.GetSymbol()}:{b_atom.GetIdx()}', f'{e_atom.GetSymbol()}:{e_atom.GetIdx()}')
    
            edge_features = np.array([])
            for edge_featurizer in edge_featurizers:
                edge_features = np.concatenate([edge_features, edge_featurizer(bond)])
    
            G.add_edge(*edge_id, edge_attr=edge_features)
    
        return G

    def _construct_simple_graph(self, mol):
        # Initialize an empty graph
        G = nx.Graph()

        # Add nodes with atom indices as node IDs
        for atom in mol.GetAtoms():
            node_id = f'{atom.GetSymbol()}:{atom.GetIdx()}'

            G.add_node(node_id, node_attr=atom.GetSymbol())

        # Add edges with bond information
        for bond in mol.GetBonds():
            b_atom = bond.GetBeginAtom()
            e_atom = bond.GetEndAtom()
            edge_id = (f'{b_atom.GetSymbol()}:{b_atom.GetIdx()}', f'{e_atom.GetSymbol()}:{e_atom.GetIdx()}')

            G.add_edge(*edge_id, edge_attr=convert_bond_type_to_string(bond))

        return G

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
            graphs = [self._construct_simple_graph(Chem.MolFromSmiles(s)) for s in smiles]
            graphs = np.array(list(graph_from_networkx(graphs, node_labels_tag=self._node_labels, edge_labels_tag=self._edge_labels)))

        else:
            graphs = [self._convert_nx_to_pyg(self._construct_features_graph(Chem.MolFromSmiles(s))) for s in smiles]
            graphs = Batch.from_data_list(graphs)

        return graphs
