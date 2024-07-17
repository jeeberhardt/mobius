#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PyRosetta
#

import numpy as np

try:
    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector, ChainSelector, ResidueIndexSelector
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
except ImportError:
    _has_pyrosetta = False
else:
    _has_pyrosetta = True


def _get_neighbour_vector(pose, residues=None, chain=None, distance=6., include_focus_in_subset=True):
    """
    Returns a vector of booleans indicating whether a residue is within a given distance from the given residues.

    Parameters
    ----------
    residues : list of str, default: None
        The list of residues in the format chainid:resid:resname.
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
    pdb2pose = pose.pdb_info().pdb2pose

    if residues is not None:
        selector = ResidueIndexSelector()

        for residue in residues:
            chainid, resid, _ = residue.split(':')
            selector.set_index(pdb2pose(chain=chainid, res=int(resid)))
    elif chain is not None:
        selector = ChainSelector(chain)
    else:
        raise RuntimeError('A list of residues or a chain must be specified')

    neighborhood_selector = NeighborhoodResidueSelector(selector, distance=distance, include_focus_in_subset=include_focus_in_subset)
    v = neighborhood_selector.apply(pose)

    return v


def _get_interface(pose, residues, distance=6.):
    """
    Returns the interface between the peptide/protein binder and the protein(s) target.

    Parameters
    ----------
    residues : list of str, default: None
        The list of residues in the format chainid:resid:resname.
    distance : float, default: 6.
        The distance cutoff.

    Returns
    -------
    interface : str
        The interface between the peptide/protein binder and the protein(s) target.

    """ 
    target_chains = []
    pose2pdb = pose.pdb_info().pose2pdb

    binder_chain = np.unique([m.split(':')[0] for m in residues])[0]

    v = _get_neighbour_vector(pose, chain=binder_chain, distance=distance)
    residue_indices = np.argwhere(list(v)).flatten()

    for idx in residue_indices:
        _, chain = pose2pdb(idx).split()
        if chain != binder_chain:
            target_chains.append(chain)

    target_chains = np.unique(target_chains)
    interface = f'{"".join(target_chains)}_{binder_chain}'

    return interface


class RosettaScorer:
    """
    A class to score mutations in peptide/protein binders using Rosetta.

    Inspiration: https://raw.githubusercontent.com/matteoferla/MEF2C_analysis/main/variant.py
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

    def __init__(self, filename, params_filenames=None):
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
        if not _has_pyrosetta:
            raise ImportError('PyRosetta is not installed.')

        options = '-no_optH false -ex1 -ex2 -mute all -beta -ignore_waters false '
        options += '-ignore_unrecognized_res true -load_PDB_components false '
        options += '-use_terminal_residues true'
        pyrosetta.init(extra_options=options)

        self.pose = pyrosetta.Pose()
        self._scorefxn = None
        self._current_mutations = None

        if params_filenames:
            params_paths = pyrosetta.rosetta.utility.vector1_string()
            params_paths.extend(params_filenames)
            pyrosetta.generate_nonstandard_residue_set(self.pose, params_paths)

        pyrosetta.rosetta.core.import_pose.pose_from_file(self.pose, filename)

    def mutate(self, mutations):
        """
        Mutate the peptide/protein binder with the given mutations list.

        Parameters
        ----------
        mutations : list of str
            The list of mutations in the format chainid:resid:resname.

        """
        pdb2pose = self.pose.pdb_info().pdb2pose

        unique_chains = np.unique([m.split(':')[0] for m in mutations])
        assert unique_chains.size == 1, f"All mutations must be within the same chain (chains: {unique_chains})."

        for mutation in mutations:
            chain, resid, new_resname = mutation.split(':')
            target_residue_idx = pdb2pose(chain=chain, res=int(resid))
            # The N and C ter residues have these extra :NtermProteinFull, blablabla suffixes
            target_residue_name = self.pose.residue(target_residue_idx).name().split(':')[0]

            assert target_residue_name in self._name3.values(), f"Residue {target_residue_name} is not a D/L standard amino acid."

            # Skip residue if it is already the same
            if target_residue_name != self._name3[new_resname]:
                residue_mutator = MutateResidue(target=target_residue_idx, new_res=self._name3[new_resname])
                residue_mutator.apply(self.pose)

        # Store mutations
        self._current_mutations = mutations

    def relax(self, distance=9., cycles=5, scorefxn="beta_relax", allow_backbone_to_move=True):
        """
        Relax the peptide/protein binder around the mutations.

        Parameters
        ----------
        distance : float, default: 9.
            The distance cutoff. The distance is from the center of mass atom, not from every atom in the molecule
        cycles : int, default: 5
            The number of relax cycles.
        scorefxn : str or `pyrosetta.rosetta.core.scoring.ScoreFunction`, default: "beta_relax"
            The name of the score function. List of suggested scoring functions:
            - beta: beta_nov16
            - beta_cart: beta_nov16_cart
            - beta_soft: beta_nov16_soft
            - franklin2019: ref2015 + dG_membrane (https://doi.org/10.1016/j.bpj.2020.03.006)
            - beta_design: beta_nov16_cart with the following weights:
                - voids_penalty: 1.0
                - hbnet: 1.0
                - hbond_sr_bb: 10.0
                - hbond_lr_bb: 10.0
                - hbond_bb_sc: 5.0
                - hbond_sc: 3.0
                - buried_unsatisfied_penalty: 0.5
            - beta_relax: beta_nov16 with the following weights:
                - arg_cation_pi: 3.0
                - approximate_buried_unsat_penalty: 5
                - approximate_buried_unsat_penalty_burial_atomic_depth: 3.5
                - approximate_buried_unsat_penalty_hbond_energy_threshold: -0.5

        """
        if not isinstance(scorefxn, pyrosetta.rosetta.core.scoring.ScoreFunction):
            if scorefxn == 'beta_design':
                scorefxn = pyrosetta.create_score_function('beta_cart')
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.voids_penalty, 1.0)
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbnet, 1.0)
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb, 10.0)
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb, 10.0)
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc, 5.0)
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc, 3.0)
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.buried_unsatisfied_penalty, 0.5)
            elif scorefxn == 'beta_relax':
                scorefxn = pyrosetta.create_score_function('beta')
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.arg_cation_pi, 3.0)
                scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.approximate_buried_unsat_penalty, 5.0)
                emo = scorefxn.energy_method_options()
                emo.approximate_buried_unsat_penalty_burial_atomic_depth(3.5)
                emo.approximate_buried_unsat_penalty_hbond_energy_threshold(-0.5)
            else:
                scorefxn = pyrosetta.create_score_function(scorefxn)

        v = _get_neighbour_vector(self.pose, residues=self._current_mutations, distance=distance)

        movemap = pyrosetta.MoveMap()
        movemap.set_bb(False)
        if allow_backbone_to_move:
            movemap.set_bb(allow_bb=v)
        movemap.set_chi(False)
        movemap.set_chi(allow_chi=v)
        movemap.set_jump(False)

        relax = FastRelax(scorefxn, cycles)
        relax.set_movemap(movemap)
        relax.set_movemap_disables_packing_of_fixed_chi_positions(True)
        if '_cart' in scorefxn.get_name():
            relax.cartesian(True)
            relax.minimize_bond_angles(True)
            relax.minimize_bond_lengths(True)
        relax.min_type('dfpmin_armijo_nonmonotone')

        relax.apply(self.pose)

    def score(self, scorefxn="beta"):
        """
        Score the peptide/protein binder.

        Parameters
        ----------
        scorefxn_name : str or `pyrosetta.rosetta.core.scoring.ScoreFunction`, default: "beta"
            The name of the score function. List of suggested scoring functions:
            - beta: beta_nov16
            - franklin2019: ref2015 + dG_membrane (https://doi.org/10.1016/j.bpj.2020.03.006)

        Returns
        -------
        scores : dict
            The scores of the peptide/protein binder with the following keys:
            - binder_energy: the energy of the peptide/protein binder
            - complex_energy: the energy of the complex
            - dG_separated: the energy difference between the complex and the separated chains
            - dSASA: the change in SASA upon complex formation
            - dG_separated/dSASAx100: the ratio between the energy and the change in SASA
            - complementary_shape: the shape complementarity, from 0 (bad) to 1 (good)
            - hbond_unsatisfied: the number of unsatisfied hydrogen bonds
            - packstat: the packing statistic, from 0 (bad) to 1 (good)

        """
        interface = _get_interface(self.pose, self._current_mutations)

        if not isinstance(scorefxn, pyrosetta.rosetta.core.scoring.ScoreFunction):
            scorefxn = pyrosetta.create_score_function(scorefxn)

        ia = InterfaceAnalyzerMover(interface)
        ia.set_compute_packstat(True)
        ia.set_scorefunction(scorefxn)
        ia.apply(self.pose)
        ia.add_score_info_to_pose(self.pose)
        data = ia.get_all_data()

        # Get peptide/protein binder energy score
        binder_chain = np.unique([m.split(':')[0] for m in self._current_mutations])[0]
        binder_energy = scorefxn.get_sub_score(self.pose, ChainSelector(binder_chain).apply(self.pose))

        results = {'binder_energy': np.around(binder_energy, decimals=3),
                   'complex_energy': np.around(ia.get_complex_energy(), decimals=3),
                   'dG_separated': np.around(ia.get_separated_interface_energy(), decimals=3),
                   'dSASA': np.around(ia.get_interface_delta_sasa(), decimals=3),
                   'dG_separated/dSASAx100': np.around(data.dG_dSASA_ratio * 100., decimals=3),
                   'complementary_shape': np.around(data.sc_value, decimals=3),
                   'hbond_unsatisfied': np.around(ia.get_interface_delta_hbond_unsat(), decimals=3),
                   'packstat': np.around(data.packstat, decimals=3)}

        # Reset mutations
        self._current_mutations = None

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
