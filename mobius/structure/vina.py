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
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdDistGeom
from rdkit.ML.Cluster import Butina
from scrubber import Scrub
from scrubber.protonate import AcidBaseConjugator
from vina import Vina


def constrained_embed_multiple_confs(query_mol, core_mol, num_confs=10, cluster=True, distance_cutoff=1.0, random_seed=0xf00d, mcs_timeout=1):
    """
    Superpose a molecule onto a reference molecule using RDKit.

    Parameters
    ----------
    query_mol : rdkit.Chem.rdchem.Mol
        The molecule to align.
    core_mol : rdkit.Chem.rdchem.Mol
        The reference molecule.
    num_confs : int, default=10
        The number of conformations to generate.
    cluster : bool, default=True
        Whether to cluster the conformations.
    distance_cutoff : float, default=1.0
        The distance cutoff to use for clustering.
    random_seed : int, default=0xf00d
        The random seed to use for the embedding.
    mcs_timeout : float, default=1
        Timeout for MCS algorithm. Helps a lot with big peptides.

    Returns
    -------
    molecules : list of rdkit.Chem.rdchem.Mol
        The list of aligned molecules, or None if no conformations 
        were generated.

    Notes
    -----
    - The superposition is done in place.
    - Source: 
        - https://github.com/forlilab/scrubber/blob/develop/scrubber/core.py#L577

    """
    # Set up the ETKDG parameters
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = random_seed
    ps.useRandomCoords = True
    ps.useBasicKnowledge = True
    ps.trackFailures = False
    ps.enforceChirality = True
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = False
    ps.clearConfs = True

    # Forcefield-related parameters
    getForceField = AllChem.UFFGetMoleculeForceField
    force_constant = 1e4
    force_tolerance = 1e-3
    energy_tolerance = 1e-4
    force_constant_final = 0.1
    max_displacement = 1.5

    # Find common substructure between probe and reference
    mcs = rdFMCS.FindMCS([query_mol, core_mol], timeout=int(mcs_timeout))
    smarts = mcs.smartsString
    patt = Chem.MolFromSmarts(smarts)
    query_match = query_mol.GetSubstructMatch(patt)
    core_match = core_mol.GetSubstructMatch(patt)
    cmap = [(i, j) for i, j in zip(query_match, core_match)]

    cids = rdDistGeom.EmbedMultipleConfs(query_mol, num_confs, ps)
    cids = list(cids)

    if len(cids) == 0:
        return None

    core_conf = core_mol.GetConformer()

    for cid in cids:
        n = 4

        # Superpose the query onto the core
        rms_ini = rdMolAlign.AlignMol(query_mol, core_mol, atomMap=cmap, prbCid=cid)

        # Do first minimization to superpose all the query atoms on the top of the core atoms
        ff = getForceField(query_mol, confId=cid)

        for atom_pair in cmap:
            p = core_conf.GetAtomPosition(atom_pair[1])
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, atom_pair[0], 0, 0, force_constant)

        ff.Initialize()
        more = ff.Minimize(energyTol=energy_tolerance, forceTol=force_tolerance)
        while more and n:
            more = ff.Minimize(energyTol=energy_tolerance, forceTol=force_tolerance)
            n -= 1

        # Do a last minimization to relax the whole query molecule without
        # deviating too far from the initial placement
        ff = getForceField(query_mol, confId=cid)

        for atom_pair in cmap:
            ff.UFFAddPositionConstraint(atom_pair[0], max_displacement, force_constant_final)

        ff.Initialize()
        ff.Minimize(energyTol=energy_tolerance, forceTol=force_tolerance)

        # Re-superpose the query onto the core
        rms_final = rdMolAlign.AlignMol(query_mol, core_mol, atomMap=cmap)

    if cluster:
        dists = []

        for i in range(len(cids)):
            for j in range(i):
                dists.append(rdMolAlign.GetBestRMS(query_mol, query_mol, i, j))

        clusts = Butina.ClusterData(dists, len(cids), distance_cutoff, isDistData=True, reordering=True)
        query_mols = [Chem.Mol(query_mol, confId=i[0]) for i in clusts]
    else:
        query_mols = [Chem.Mol(query_mol, confId=i) for i in cids]

    return query_mols


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

        """
        self._vina = Vina(verbosity=0)
        self._vina.set_receptor(receptor_pdbqt_filename)
        self._vina.compute_vina_maps(center, dimensions)
        self._protonator = AcidBaseConjugator.from_default_data_files()
        self._preparator = MoleculePreparation()

    def dock(self, input_molecule, template=None, num_confs=10, cluster=True, adjust_protonation=True, pH=7.4):
        """
        Superpose input molecule onto a reference molecule, if provided. 
        If not, do global docking.

        Parameters
        ----------
        input_molecule : rdkit.Chem.rdchem.Mol
            The input molecule to be docked.
        template : rdkit.Chem.rdchem.Mol, default=None
            The template molecule to use for the superposition. If None,
            the input molecule will be docked (global search).
        num_confs: int, default=10
            Number of conformation generated in total, if reference provided.
        cluster : bool, default=True
            Cluster conformations before scoring, if reference provided.
            Use Butina clustering method with a distance cutoff of 1 A.
        adjust_protonation : bool, default=True
            Whether the protonation of the input molecule is adjusted 
            (using Scrubber) or not before docking. If not, the input molecule 
            is expected have the correct protonation.
        pH : float, default=7.4
            The pH to use if the protonation of the input molecule is adjusted 
            before docking.

        Returns
        -------
        docked_molecule : rdkit.Chem.rdchem.Mol
            The RDKit molecule of the docked molecule, or None if the docking failed.
        scores : np.array
            The Vina scores of the docked molecule.

        """
        output_pdbqt = 'docker_tmp.pdbqt'

        if adjust_protonation:
            input_molecule = self._protonator(input_molecule, pH, pH)[0]

        if template is not None:
            all_scores = []
            all_pdbqt_mols = []

            # Ignore warning because hydrogen atoms are missing
            RDLogger.DisableLog('rdApp.warning')

            mols = constrained_embed_multiple_confs(input_molecule, template, num_confs=num_confs, cluster=cluster)
                
            RDLogger.EnableLog('rdApp.warning')

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
