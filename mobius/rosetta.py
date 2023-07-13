#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PyRosetta
#

import os
import joblib

import numpy as np
import pandas as pd
import pyrosetta
import ray
from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector, ChainSelector, ResidueIndexSelector
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

from . import utils


class ProteinPeptideComplex:
    """
    A class to handle protein-peptide complexes.

    Original source: https://raw.githubusercontent.com/matteoferla/MEF2C_analysis/main/variant.py
    """

    _name3 = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 
              'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
              'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
              'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
              'dA': 'DALA', 'dC': 'DCYS', 'dD': 'DASP', 'dE': 'DGLU', 
              'dF': 'DPHE', 'dH': 'DHIS', 'dI': 'DILE', 'dL': 'DLEU', 
              'dK': 'DLYS', 'dM': 'DMET', 'dN': 'DASN', 'dP': 'DPRO', 
              'dQ': 'DGLN', 'dR': 'DARG', 'dS': 'DSER', 'dT': 'DTHR', 
              'dV': 'DVAL', 'dW': 'DTRP', 'dY': 'DTYR'}

    def __init__(self, filename, chain, params_filenames=None):
        """
        Loads a pose from filename with the params in the params_folder

        Parameters
        ----------
        filename : str
            The filename of the pdb file.
        chain : str
            The chain id of the peptide.
        params_filenames : list of str, default: None
            The filenames of the params files.

        """
        options = '-no_optH false -ex1 -ex2 -mute all -beta -ignore_unrecognized_res true -load_PDB_components false -ignore_waters false'
        pyrosetta.init(extra_options=options)
        
        self.pose = pyrosetta.Pose()
        self._peptide_chain = chain
        self._scorefxn = None

        if params_filenames:
            params_paths = pyrosetta.rosetta.utility.vector1_string()
            params_paths.extend(params_filenames)
            pyrosetta.generate_nonstandard_residue_set(self.pose, params_paths)

        pyrosetta.rosetta.core.import_pose.pose_from_file(self.pose, filename)

    def _get_neighbour_vector(self, residues=None, chain=None, distance=6., include_focus_in_subset=True):
        """
        Returns a vector of booleans indicating whether a residue is within a given distance from the residues or chain.

        A list of residues or a chain must be specified.

        Parameters
        ----------
        residues : list of str, default: None
            The list of residues in the format chainid:resid.
        chain : str, default: None
            The chain id.
        distance : float, default: 6.
            The distance cutoff.
        include_focus_in_subset : bool, default: True
            Whether to include the residues or chain in the vector.

        Returns
        -------
        v : pyrosetta.rosetta.utility.vector1_bool
            The vector of booleans.

        """
        pdb2pose = self.pose.pdb_info().pdb2pose

        if residues is not None:
            selector = ResidueIndexSelector()

            for residue in residues:
                chainid, resid = residue.split(':')[:2]
                selector.set_index(pdb2pose(chain=chainid, res=int(resid)))
        elif chain is not None:
            selector = ChainSelector(chain)
        else:
            raise RuntimeError('Either residues or chain must be specified')

        neighborhood_selector = NeighborhoodResidueSelector(selector, distance=distance, include_focus_in_subset=include_focus_in_subset)
        v = neighborhood_selector.apply(self.pose)

        return v

    def relax_peptide(self, distance=9., cycles=5, scorefxn="beta_cart"):
        """
        Relaxes the peptide chain.

        Parameters
        ----------
        distance : float, default: 9.
            The distance cutoff. The distance is from the center of mass atom, not from every atom in the molecule
        cycles : int, default: 5
            The number of relax cycles.
        scorefxn : str or `pyrosetta.rosetta.core.scoring.ScoreFunction`, default: "beta_cart"
            The name of the score function. List of suggested scoring functions:
            - beta_cart: beta_nov16_cart
            - beta_soft: beta_nov16_soft
            - franklin2019: ref2015 + dG_membrane (https://doi.org/10.1016/j.bpj.2020.03.006)

        """
        if not isinstance(scorefxn, pyrosetta.rosetta.core.scoring.ScoreFunction):
            scorefxn = pyrosetta.create_score_function(scorefxn)

        v = self._get_neighbour_vector(chain=self._peptide_chain, distance=distance)

        movemap = pyrosetta.MoveMap()
        movemap.set_bb(False)
        movemap.set_bb(allow_bb=v)
        movemap.set_chi(False)
        movemap.set_chi(allow_chi=v)
        movemap.set_jump(False)

        relax = FastRelax(scorefxn, cycles)
        relax.set_movemap(movemap)
        relax.set_movemap_disables_packing_of_fixed_chi_positions(True)
        if '_cart' in scorefxn.get_name():
            relax.cartesian(True)
        relax.min_type('dfpmin_armijo_nonmonotone')
        relax.apply(self.pose)

    def mutate(self, mutations):
        """
        Mutates the peptide with the given mutations list.

        Parameters
        ----------
        mutations : list of str
            The list of mutations in the format resid:resname.

        """
        pdb2pose = self.pose.pdb_info().pdb2pose

        for mutation in mutations:
            resid, new_resname = mutation.split(':')
            target_residue_idx = pdb2pose(chain=self._peptide_chain, res=int(resid))
            # The N and C ter residues have these extra :NtermProteinFull, blablabla suffixes
            target_residue_name = self.pose.residue(target_residue_idx).name().split(':')[0]

            assert target_residue_name in self._name3.values(), f"Residue {target_residue_name} is not a D/L standard amino acid."

            # Skip residue if it is already the same
            if target_residue_name != self._name3[new_resname]:
                residue_mutator = MutateResidue(target=target_residue_idx, new_res=self._name3[new_resname])
                residue_mutator.apply(self.pose)
    
    def _get_interface(self, peptide_chain):
        """
        Returns the interface between the peptide and the protein.

        Parameters
        ----------
        peptide_chain : str
            The chain id of the peptide.
        
        Returns
        -------
        interface : str
            The interface between the peptide and the protein.

        """ 
        protein_chains = []
        pose2pdb = self.pose.pdb_info().pose2pdb

        v = self._get_neighbour_vector(chain=peptide_chain, distance=9.)
        residue_indices = np.argwhere(list(v)).flatten()

        for idx in residue_indices:
            _, chain = pose2pdb(idx).split()
            if chain != peptide_chain:
                protein_chains.append(chain)
        
        protein_chains = np.unique(protein_chains)
        interface = f'{"".join(protein_chains)}_{peptide_chain}'
        
        return interface

    def score(self, scorefxn="beta"):
        """
        Scores peptide.

        Parameters
        ----------
        scorefxn_name : str or `pyrosetta.rosetta.core.scoring.ScoreFunction`, default: "beta"
            The name of the score function. List of suggested scoring functions:
            - beta: beta_nov16
            - franklin2019: ref2015 + dG_membrane (https://doi.org/10.1016/j.bpj.2020.03.006)

        Returns
        -------
        scores : dict
            The scores of the peptide with the following keys:
            - dG_separated: the energy difference between the complex and the separated chains
            - dSASA: the change in SASA upon complex formation
            - dG_separated/dSASAx100: the ratio between the energy and the change in SASA
            - complementary_shape: the shape complementarity
            - hbond_unsatisfied: the number of unsatisfied hydrogen bonds

        """
        interface = self._get_interface(self._peptide_chain)

        if not isinstance(scorefxn, pyrosetta.rosetta.core.scoring.ScoreFunction):
            scorefxn = pyrosetta.create_score_function(scorefxn)

        ia = InterfaceAnalyzerMover(interface)
        ia.set_scorefunction(scorefxn)
        ia.apply(self.pose)
        data = ia.get_all_data()

        results = {'dG_separated': np.around(ia.get_separated_interface_energy(), decimals=3),
                   'dSASA': np.around(ia.get_interface_delta_sasa(), decimals=3),
                   'dG_separated/dSASAx100': np.around(data.dG_dSASA_ratio * 100., decimals=3),
                   'complementary_shape': np.around(data.sc_value, decimals=3),
                   'hbond_unsatisfied': np.around(ia.get_interface_delta_hbond_unsat(), decimals=3)}

        return results

    def export_pdb(self, output_filename):
        """
        Exports the pose to a pdb file.

        Parameters
        ----------
        output_filename : str
            The output filename.

        """
        self.pose.dump_pdb(output_filename)


def _polymer_to_mutations(polymer):
    """
    Converts polymer in HELM format into a list of mutations.

    Parameters
    ----------
    polymer : str
        The polymer in HELM format.

    Returns
    -------
    mutations : list of str
        The list of mutations in the format resid:resname.

    """
    mutations = {}

    complex_polymer, _, _, _ = utils.parse_helm(polymer)

    for pid, residues in complex_polymer.items():
        for i, resname in enumerate(residues, start=1):
            mutation = f"{i}:{resname}"
            mutations.setdefault(pid, []).append(mutation)

    return mutations


class ProteinPeptideScorer:
    def __init__(self, pdb_filename, chain, params_filenames=None, n_process=-1):
        """
        A class to score peptides in parallel.
        
        Parameters
        ----------
        pdb_filename : str
            The filename of the pdb file.
        chain : str
            The chain id of the peptide.
        params_filenames : list of str, default: None
            The filenames of the params files.
        n_process : int, default: -1
            The number of processes to use. If -1, use all available CPUs.
        
        """
        self._filename = pdb_filename
        self._params_filenames = params_filenames
        self._peptide_chain = chain
        self._n_process = n_process
    
    @staticmethod
    @ray.remote
    def process_peptide(peptide, filename, peptide_chain, params_filenames, distance, cycles, scorefxn):
        """
        Processes a peptide.
        
        Parameters
        ----------
        peptide : str
            The peptide in HELM format.
        filename : str
            The filename of the pdb file.
        peptide_chain : str
            The chain id of the peptide.
        params_filenames : list of str, default: None
            The filenames of the params files.
        distance : float, default: 9.
            The distance cutoff. The distance is from the center of mass atom, not from every atom in the molecule
        cycles : int, default: 5
            The number of relax cycles.
        scorefxn : str or `pyrosetta.rosetta.core.scoring.ScoreFunction`, default: "beta_cart"
            The name of the score function. List of suggested scoring functions:
            - beta_cart: beta_nov16_cart
            - beta_soft: beta_nov16_soft
            - franklin2019: ref2015 + dG_membrane (https://doi.org/10.1016/j.bpj.2020.03.006)
            
        Returns
        -------
        pdb_string : str
            The PDB file of the protein-peptide complex as string.
        peptide : str
            The peptide in HELM format.
        scores : dict
            The scores of the peptide with the following keys:
                
        """
        cplex = ProteinPeptideComplex(filename, peptide_chain, params_filenames)

        # Transform HELM into a list of mutations
        mutations = _polymer_to_mutations(peptide)

        assert len(mutations) == 1, "Peptide with more than one polymer chain is not supported."

        cplex.mutate(list(mutations.values())[0])
        cplex.relax_peptide(distance=distance, cycles=cycles, scorefxn=scorefxn)
        scores = cplex.score()
        
        # Write PDB into a string
        buffer = pyrosetta.rosetta.std.stringbuf()
        cplex.pose.dump_pdb(pyrosetta.rosetta.std.ostream(buffer))
        pdb_string = buffer.str()

        return pdb_string, peptide, scores
    
    def score_peptides(self, peptides, distance=9., cycles=5, scorefxn="beta_cart"):
        """
        Scores peptides in parallel.

        Parameters
        ----------
        peptides : list of str
            The list of peptides in HELM format.
        distance : float, default: 9.
            The distance cutoff. The distance is from the center of mass atom, not from every atom in the molecule.
        cycles : int, default: 5
            The number of relax cycles.
        scorefxn : str or `pyrosetta.rosetta.core.scoring.ScoreFunction`, default: "beta_cart"
            The name of the score function. List of suggested scoring functions:
            - beta_cart: beta_nov16_cart
            - beta_soft: beta_nov16_soft
            - franklin2019: ref2015 + dG_membrane (https://doi.org/10.1016/j.bpj.2020.03.006)

        Returns
        -------
        pdbs : list of str
            The list of PDB files of the protein-peptide complexes as strings.
        df : pandas.DataFrame
            The dataframe containing the scores of the peptides.

        """
        data = []
        pdbs = []
        
        if self._n_process == -1:
            # joblib.cpu_count() returns the exact number available that can be used on the cluster
            # unlike os.cpu_count, ray.init() or multiprocessing.cpu_count()...
            n_process = np.min([joblib.cpu_count(), len(peptides)])
        else:
            n_process = self._n_process
        
        ray.init(num_cpus=int(n_process), ignore_reinit_error=True)
        
        # Process peptides in parallel
        refs = [self.process_peptide.remote(peptide, 
                                            self._filename, self._peptide_chain, self._params_filenames, 
                                            distance, cycles, scorefxn) for peptide in peptides]
        results = ray.get(refs)
        
        # Collect results
        for r in results:
            pdbs.append(r[0])
            data.append([r[1]] + list(r[2].values()))

        columns = ['peptide'] + list(results[0][2].keys())
        df = pd.DataFrame(data=data, columns=columns)
        
        ray.shutdown()
        
        return pdbs, df
