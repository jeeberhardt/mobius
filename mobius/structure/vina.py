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
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
from scrubber import Scrub
from vina import Vina


def constrained_embed_multiple_confs(mobile, ref, random_seed=0xf00d, num_confs=10, cluster=True, distance_cutoff=1.0):
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
    - Sources: 
        - https://greglandrum.github.io/rdkit-blog/posts/2023-02-10-more-on-constrained-embedding.html
        - https://github.com/rdkit/rdkit/issues/3266

    """
    max_displacement = 0.5
    force_constant = 1.e4

    mcs = rdFMCS.FindMCS([ref, mobile])
    smarts = mcs.smartsString
    patt = Chem.MolFromSmarts(smarts)
    mobile_match = mobile.GetSubstructMatch(patt)
    ref_match = ref.GetSubstructMatch(patt)

    cmap = {mobile_match[i]: ref.GetConformer().GetAtomPosition(ref_match[i]) for i in range(len(ref_match))}
    cids = AllChem.EmbedMultipleConfs(mobile, numConfs=num_confs, coordMap=cmap, randomSeed=random_seed, 
                                      useRandomCoords=True, useBasicKnowledge=False, enforceChirality=False)

    cids = list(cids)

    if len(cids) == 0:
        return None

    # Minimize conformations since we set useBasicKnowledge to False,
    # but we minimize only the atoms that were not matched.
    mp = AllChem.MMFFGetMoleculeProperties(mobile)

    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(mobile, mp, confId=cid)
        for i in mobile_match:
            ff.MMFFAddPositionConstraint(i, max_displacement, force_constant)
        ff.Minimize()

    if cluster:
        dists = []

        for i in range(len(cids)):
            for j in range(i):
                dists.append(rdMolAlign.GetBestRMS(mobile, mobile, i, j))

        clusts = Butina.ClusterData(dists, len(cids), distance_cutoff, isDistData=True, reordering=True)
        mobiles = [Chem.Mol(mobile, confId=i[0]) for i in clusts]
    else:
        mobiles = [Chem.Mol(mobile, confId=i) for i in cids]

    return mobiles


class VinaScorer:

    def __init__(self, receptor_pdbqt_filename, center, dimensions):
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
        scrub : bool, default=True
            Whether scrub the input molecule (using Scrubber) or not before docking. 
            If not, the input molecule is expected to be fully prepared (hydrogen, protonations, ...)
        pH : float, default=7.4
            The pH to use if the molecule is scrubed before docking.

        """
        self._vina = Vina(verbosity=0)
        self._vina.set_receptor(receptor_pdbqt_filename)
        self._vina.compute_vina_maps(center, dimensions)
        self._preparator = MoleculePreparation()

    def dock(self, input_molecule, reference=None, scrub=True, pH=7.4, num_confs=10, cluster=True):
        """
        Superpose input molecule onto a reference molecule, if provided. 
        If not, do global docking.

        Parameters
        ----------
        input_molecule : rdkit.Chem.rdchem.Mol
            The input molecule to be docked.
        reference : rdkit.Chem.rdchem.Mol, default=None
            The reference molecule to use for the superposition. If None,
            the molecule will be docked.
        scrub : bool, default=True
            Whether scrub the input molecule (using Scrubber) or not before docking. 
            If not, the input molecule is expected to be fully prepared (hydrogen, protonations, ...)
        pH : float, default=7.4
            The pH to use if the molecule is scrubed before docking.
        num_confs: int, default=10
            Number of conformation to generate, if reference provided.
        cluster : bool, default=True
            Cluster conformations before scoring, if reference provided.
            Use Butina clustering method with a distance cutoff of 1 A.

        Returns
        -------
        docked_molecule : rdkit.Chem.rdchem.Mol
            The RDKit molecule of the docked molecule.
        scores : np.array
            The Vina scores of the docked molecule.

        """
        output_pdbqt = 'docker_tmp.pdbqt'

        if scrub:
            scrubber = Scrub(ph_low=pH)
            input_molecule = scrubber(input_molecule)[0]

        if reference is not None:
            all_scores = []
            all_pdbqt_mols = []

            mols = constrained_embed_multiple_confs(input_molecule, reference, num_confs=num_confs, cluster=cluster)

            if mols is None:
                return None, np.array([np.nan] * 8)

            for mol in mols:
                mol_setup = self._preparator.prepare(mol)[0]
                pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]

                try:
                    self._vina.set_ligand_from_string(pdbqt_string)
                except TypeError:
                    print('Could not read input PDBQT file!')
                    return None, np.array([np.nan] * 8)

                # Do local optimization
                self._vina.optimize()
                all_scores.append(self._vina.score())
                self._vina.write_pose(output_pdbqt, overwrite=True)
                all_pdbqt_mols.append(PDBQTMolecule.from_file(output_pdbqt, is_dlg=False, skip_typing=True))

            # Keep only best pose based on total energy
            best_mol_id = np.argmin(np.asarray(all_scores)[:, 0])
            scores = all_scores[best_mol_id]
            pdbqt_mol = all_pdbqt_mols[best_mol_id]

            # Cleanup temp pdbqt file
            os.remove(output_pdbqt)
        else:
            mol_setup = self._preparator.prepare(input_molecule)[0]
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
