#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PyRosetta
#

import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector, ChainSelector, ResidueIndexSelector
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover


class ProteinPeptideComplex:
    """
    A class to handle protein-peptide complexes.

    Original source: https://raw.githubusercontent.com/matteoferla/MEF2C_analysis/main/variant.py
    """

    _name3 = {'A': 'ALA',
              'C': 'CYS',
              'D': 'ASP',
              'E': 'GLU',
              'F': 'PHE',
              'G': 'GLY',
              'H': 'HIS',
              'I': 'ILE',
              'L': 'LEU',
              'K': 'LYS',
              'M': 'MET',
              'N': 'ASN',
              'P': 'PRO',
              'Q': 'GLN',
              'R': 'ARG',
              'S': 'SER',
              'T': 'THR',
              'V': 'VAL',
              'W': 'TRP',
              'Y': 'TYR'}

    def __init__(self, filename, params_filenames=None):
        """
        Loads a pose from filename with the params in the params_folder

        Parameters
        ----------
        filename : str
            The filename of the pdb file.
        params_filenames : list of str, default: None
            The filenames of the params files.

        """
        options = '-no_optH false -ex1 -ex2 -mute all -beta -ignore_unrecognized_res true -load_PDB_components false -ignore_waters false'
        pyrosetta.init(extra_options=options)
        self.pose = self.load_pose_from_file(filename, params_filenames)
        self._scorefxn = None

    def load_pose_from_file(self, filename, params_filenames=None):
        """
        Loads a pose from filename with the params in the params_folder

        Parameters
        ----------
        filename : str
            The filename of the pdb file.
        params_filenames : list of str, default: None
            The filenames of the params files.

        """
        pose = pyrosetta.Pose()

        if params_filenames:
            params_paths = pyrosetta.rosetta.utility.vector1_string()
            params_paths.extend(params_filenames)
            pyrosetta.generate_nonstandard_residue_set(pose, params_paths)

        pyrosetta.rosetta.core.import_pose.pose_from_file(pose, filename)

        return pose

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

    def relax_peptide(self, chain, distance=6., cycles=5, scorefxn="beta_cart"):
        """
        Relaxes the peptide chain.

        Parameters
        ----------
        chain : str
            The chain id of the peptide.
        distance : float, default: 6.
            The distance cutoff.
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

        v = self._get_neighbour_vector(chain=chain, distance=distance)

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
        Mutates the residues.

        Parameters
        ----------
        mutations : list of str
            The list of mutations in the format chainid:resid:resname.

        """
        pdb2pose = self.pose.pdb_info().pdb2pose

        for mutation in mutations:
            chainid, resid, new_resname = mutation.split(':')
            target_residue_idx = pdb2pose(chain=chainid, res=int(resid))
            # The N and C ter residues have these extra :NtermProteinFull, blablabla suffixes
            target_residue_name = self.pose.residue(target_residue_idx).name().split(':')[0]

            if target_residue_name != self._name3[new_resname]:
                residue_mutator = MutateResidue(target=target_residue_idx, new_res=self._name3[new_resname])
                residue_mutator.apply(self.pose)

    def _has_interface(self, interface):
        """
        Checks whether the pose has the interface.

        Parameters
        ----------
        interface : str
            The interface name (e.g. AB_C, which means that the interface is between chains AB and chain C).

        Returns
        -------
        has_interface : bool
            Whether the pose has the interface or not.

        """
        pose2pdb = self.pose.pdb_info().pose2pdb
        have_chains = {pose2pdb(r).split()[1] for r in range(1, self.pose.total_residue() + 1)}
        want_chains = set(interface.replace('_', ''))

        return have_chains == want_chains

    def score_interface(self, interface, scorefxn="beta"):
        """
        Scores the interface.

        Parameters
        ----------
        interface : str
            The interface name (e.g. AB_C, which means that the interface is between chains AB and chain C).
        scorefxn_name : str or `pyrosetta.rosetta.core.scoring.ScoreFunction`, default: "beta"
            The name of the score function. List of suggested scoring functions:
            - beta: beta_nov16
            - franklin2019: ref2015 + dG_membrane (https://doi.org/10.1016/j.bpj.2020.03.006)

        Returns
        -------
        scores : dict
            The scores of the interface with the following keys:
            - dG_separated: the energy difference between the complex and the separated chains
            - dSASA: the change in SASA upon complex formation
            - dG_separated/dSASAx100: the ratio between the energy and the change in SASA
            - complementary_shape: the shape complementarity
            - hbond_unsatisfied: the number of unsatisfied hydrogen bonds

        """
        assert self._has_interface(interface), f'There is no {interface}'

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
