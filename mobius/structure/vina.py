#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# mobius - vina
#


import os

import numpy as np
from meeko import PDBQTMolecule
from meeko import MoleculePreparation
from meeko import RDKitMolCreate
from meeko import PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from scrubber import Scrub
from vina import Vina


def superpose_molecule(mobile, ref, random_seed=0xf00d):
    """
    Superpose a molecule onto a reference molecule using RDKit.

    Parameters
    ----------
    mobile : rdkit.Chem.rdchem.Mol
        The molecule to align.
    ref : rdkit.Chem.rdchem.Mol
        The reference molecule.
    random_seed : int, default=0xf00d
        The random seed to use for the embedding.

    Returns
    -------
    success : bool
        Whether the superposition was successful.
    rmsd_core : float
        The RMSD of the aligned core.

    Notes
    -----
    - The superposition is done in place.
    - Source: https://greglandrum.github.io/rdkit-blog/posts/2023-02-10-more-on-constrained-embedding.html

    """
    delta = 0.0
    max_displacement = 0.5
    force_constant = 1.e4

    # Get common substructure between mobile and ref molecules
    mcs = rdFMCS.FindMCS([ref, mobile])
    smarts = mcs.smartsString
    patt = Chem.MolFromSmarts(smarts)
    mobile_match = mobile.GetSubstructMatch(patt)
    ref_match = ref.GetSubstructMatch(patt)

    # Do the superposition
    cmap = {mobile_match[i]:ref.GetConformer().GetAtomPosition(ref_match[i]) for i in range(len(ref_match))}
    failed = AllChem.EmbedMolecule(mobile, randomSeed=random_seed, coordMap=cmap, useRandomCoords=True, 
                                   useBasicKnowledge=False, enforceChirality=False)
    
    if failed:
        return False, np.nan
    
    # Minimize conformation since we set useBasicKnowledge to False,
    # but we minimize only the atoms that were not matched.
    mp = AllChem.MMFFGetMoleculeProperties(mobile)
    ff = AllChem.MMFFGetMoleculeForceField(mobile, mp)
    for i in mobile_match:
        ff.MMFFAddPositionConstraint(i, max_displacement, force_constant)
    ff.Minimize()

    # Compute RMSD between matched atoms
    for ref_i, mobile_i in zip(ref_match, mobile_match):
        delta += (ref.GetConformer().GetAtomPosition(ref_i) - mobile.GetConformer().GetAtomPosition(mobile_i)).LengthSq()
    rmsd_core = np.sqrt(delta / len(mobile_match))

    return True, rmsd_core


class VinaScorer:

    def __init__(self, receptor_pdbqt_filename, center, dimensions, protonate=True, pH=7.4):
        """
        Constructs a new instance of the Vina scorer.

        Parameters
        ----------
        receptor_pdbqt_filename : str
            The filename of the receptor PDBQT file.
        center : list
            The center of the docking box.
        dimensions : list
            The dimensions of the docking box.
        protonate : bool, default=True
            Whether to protonate the input ligand before docking.
        pH : float, default=7.4
            The pH to use for protonation, if protonation is enabled.

        """
        self._vina = Vina(verbosity=0)
        self._vina.set_receptor(receptor_pdbqt_filename)
        self._vina.compute_vina_maps(center, dimensions)
        self._protonate = protonate
        self._scrubber = Scrub(ph_low=pH)
        self._preparator = MoleculePreparation()

    def dock(self, smiles, reference=None):
        """
        Docks a molecule on a reference molecule.

        Parameters
        ----------
        smiles : str
            The SMILES of the molecule to dock.
        reference : rdkit.Chem.rdchem.Mol, default=None
            The reference molecule to use for the superposition. If None,
            the molecule will be docked.

        Returns
        -------
        molecule : rdkit.Chem.rdchem.Mol
            RDKit molecule of the docked molecule.
        scores : np.array
            Vina scores of the docked molecule.

        """
        output_pdbqt = 'docker_tmp.pdbqt'

        mobile = Chem.MolFromSmiles(smiles)

        if self._protonate:
            mobile = self._scrubber(mobile)[0]
        else:
            mobile = Chem.AddHs(mobile)
        
        if mobile is None:
            return None, np.array([np.nan] * 8)
        
        if reference is not None:
            succeeded, _ = superpose_molecule(mobile, reference, random_coordinates=True)

            if not succeeded:
                return None, np.array([np.nan] * 8)

            mol_setup = self._preparator.prepare(mobile)[0]
            pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]

            try:
                self._vina.set_ligand_from_string(pdbqt_string)
            except TypeError:
                print('Could not read input PDBQT file!')
                return None, np.array([np.nan] * 8)

            # Do local optimization
            self._vina.optimize()
            scores = self._vina.score()
            self._vina.write_pose(output_pdbqt, overwrite=True)
            pdbqt_mol = PDBQTMolecule.from_file(output_pdbqt, is_dlg=False, skip_typing=True)
            os.remove(output_pdbqt)
        else:
            mol_setup = self._preparator.prepare(mobile)[0]
            pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]

            try:
                self._vina.set_ligand_from_string(pdbqt_string)
            except TypeError:
                print('Could not read input PDBQT file!')
                return None, np.array([np.nan] * 8)

            # Do global optimization
            self._vina.dock()
            scores = self._vina.energies(n_poses=1)
            output_pdbqt = self._vina.poses(n_poses=1)
            pdbqt_mol = PDBQTMolecule(output_pdbqt)

        mol_docked = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0]

        return mol_docked, scores
