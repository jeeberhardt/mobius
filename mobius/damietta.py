#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Damietta
#

import contextlib
import glob
import os
import shlex
import shutil
import subprocess
import tempfile

import numpy as np
import parmed as pmd
from pdbfixer.pdbfixer import PDBFixer
from openmm.app import ForceField, NoCutoff, HBonds, Simulation
from openmm import VerletIntegrator, unit, Platform
from prody import parsePDB, writePDB


@contextlib.contextmanager
def temporary_directory(suffix=None, prefix=None, dir=None, clean=True):
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yield temp_dir
    finally:
        os.chdir(cwd)
        if clean:
            shutil.rmtree(temp_dir)


def execute_command(cmd_line):
    """Simple function to execute bash command."""
    args = shlex.split(cmd_line)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, errors = p.communicate()

    return output, errors


def run_sinai_cs_f2m2f(spec_input_filename, damietta_path):
    command_line = f'{damietta_path}/bin/sinai_cs_f2m2f {spec_input_filename} '
    outputs, _ = execute_command(command_line)

    if 'failed to parse combinatorial sampling (cs) specs file' in outputs:
        raise RuntimeError(f'Command: {command_line} failed with error message: {outputs}')
    elif 'internal/library input error' in outputs:
        raise RuntimeError(f'Command: {command_line} failed with error message: {outputs}')


def format_pdb_for_damietta(input_pdb_filename, output_pdb_filename):
    i = 1
    new_pdb = ''
    atom_str = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}		  {:>2s}{:2s}\n"

    # Mapping between Amber atom names and CHARMM atom names
    amber_to_charmm_atom_types = {
        'ALA': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN'},
        'ARG': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HD3': 'HD1', 'HG3': 'HG1'},
        'ASN': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1'},
        'ASP': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1'},
        'CYS': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HG': 'HG1'},
        'GLN': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HG3': 'HG1'},
        'GLU': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HG3': 'HG1'},
        'GLY': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HA3': 'HA1'},
        'HIS': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1'},
        'ILE': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HG13': 'HG11', 'CD1': 'CD', 'HD11': 'HD1', 'HD12': 'HD2', 'HD13': 'HD3'},
        'LEU': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1'},
        'LYS': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HG3': 'HG1', 'HD3': 'HD1', 'HE3': 'HE1'},
        'MET': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HG3': 'HG1'},
        'PHE': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1'},
        'PRO': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HD3': 'HD1', 'HG3': 'HG1'},
        'SER': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1', 'HG': 'HG1'},
        'THR': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN'},
        'TRP': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1'},
        'TYR': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', 'HB3': 'HB1'},
        'VAL': {'H1': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'OXT': 'OT2', 'H': 'HN', },
        }

    pdb = parsePDB(input_pdb_filename)

    for atom in pdb.iterAtoms():
        atom_name = atom.getName()
        resname = atom.getResname()
        x, y, z = atom.getCoords()

        if resname == 'HSD' or resname == 'HSE' or resname == 'HSP':
            resname = 'HIS'
        
        # Convert Amber atom names to CHARMM atom names, if needed
        if atom_name in amber_to_charmm_atom_types[resname]:
            try:
                atom_name = amber_to_charmm_atom_types[resname][atom_name]
            except KeyError:
                pass

        if atom_name == 'OT1':
            atom_name = 'O'
        elif atom_name == 'OT2':
            # Damietta does not like OT2 atom type, so we skip it
            continue

        new_pdb += atom_str.format('ATOM', i, atom_name, ' ', resname, atom.getChid(), atom.getResnum(), ' ', x, y, z, 1.0, 0.0, ' ', ' ')

        i += 1

    with open(output_pdb_filename, 'w') as w:
        w.write(new_pdb)


class DamiettaScorer:
    """
    A class to score mutations in peptide/protein binders using Damietta.

    """

    def __init__(self, pdb_filename, damietta_path):
        """
        Loads a pose from filename with the params in the params_folder

        Parameters
        ----------
        filename : str
            The filename of the pdb file, prepared for Damietta.
        damietta_path : str
            The path of the Damietta directory.

        Notes
        -----
        - The input pdb file must be fully protonated. Use REDUCE to add hydrogen atoms, as the energy 
        minimization is performed using the Amber force field.

        """
        self._pdb_filename = pdb_filename
        self._damietta_path = os.path.abspath(damietta_path)
        self._damietta_lib_path = None

        # Locate damietta library in the path
        directories = glob.glob(damietta_path + '/*/')
        for directory in directories:
            if 'libv' in directory:
                self._damietta_lib_path = os.path.abspath(directory)
                break

        if self._damietta_lib_path is None:
            raise RuntimeError('Could not locate Damietta library in {self._damietta_path} directory.')

        self._pdb = parsePDB(self._pdb_filename)
        self._mutated_pdb = None
        self._residue_to_indices = {f'{r.getChid()}:{r.getResnum()}': i + 1 for i, r in enumerate(self._pdb.iterResidues())}

    def minimize(self, max_iterations=100, platform='CPU', clean=True):
        """
        Minimizes the protein/peptide.

        Parameters
        ----------
        max_iterations : int, default: 100
            The maximum number of iterations for the minimization.
        platform : str, default: 'CPU'
            The platform to use for running the minimization ('CPU', 'CUDA' or 'OpenCL').
        clean : bool, default: True
            Whether to clean the temporary files. Default is True.

        Notes
        -----
        - The minimization is performed using the Amber ff14SB force field and the GBn2 implicit solvent model,
        instead of the CHARMM forcefield used by Damietta internally, for convenience.


        """
        input_pdb_filename = 'protein.pdb'
        minimized_pdb_filename = 'minimized.pdb'

        assert platform in ['CPU', 'CUDA', 'OpenCL'], 'Platform must be either CPU, CUDA or OpenCL.'

        with temporary_directory(prefix='dm_min_', dir='.', clean=clean) as tmp_dir:
            writePDB(input_pdb_filename, self._pdb, renumber=False)

            # PDB is going to be fixed only if we already mutated residues with damietta
            pdb = PDBFixer(filename=input_pdb_filename)
            pdb.findMissingResidues()
            pdb.findMissingAtoms()
            pdb.addMissingAtoms()

            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

            system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
            integrator = VerletIntegrator(0.001*unit.picoseconds)

            if platform != 'CPU':
                properties = {"Precision": "mixed"}
            else:
                properties = {}

            platform = Platform.getPlatformByName(platform)
            simulation = Simulation(pdb.topology, system, integrator, platform, properties)
            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy(maxIterations=max_iterations)

            # Get new coordinates, and replace ones in prody structure
            state = simulation.context.getState(getEnergy=True, getPositions=True)
            new_pdb = pmd.openmm.load_topology(pdb.topology, system, xyz=state.getPositions())
            new_pdb.save(minimized_pdb_filename, overwrite=True)

            # Replace the pdb with the minimized one
            self._pdb = parsePDB(minimized_pdb_filename)
            #self._pdb.setCoords(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))

    def mutate_score(self, mutations, repack_neighbor_residues=False, neighbor_cutoff=4.0, load_memory=True, clean=True):
        """
        Mutates the protein/peptide with the given mutations list.

        Parameters
        ----------
        mutations : list of str
            The list of mutations in the format chainid:resid:resname.
        repack_neighbor_residues : bool, default: False
            Whether to repack the residues within the given cutoff distance of the mutated residues.
        neighbor_cutoff : float, default: 4.0
            The cutoff distance to consider the neighboring residues.
        load_memory : bool, default: True
            Whether to load the memory.
        clean : bool, default: True
            Whether to clean the temporary files.

        Returns
        -------
        energies : ndarray of float
            The energies (in kcal/mol) of the mutated protein. The energies returned
            are in the following order: (pp_dG, k_dG, lj, solv, elec, total).

        Notes
        -----
        - The first (N-terminal) and the last (C-terminal) residues of the protein can not be mutated, 
        since either phi or psi dihedral angle is not defined for them.

        - The current version (v1.95) of the Damietta toolkit does not account for any interactions with 
        heteroatoms (e.g. ligands, cofactors, ions, solvent molecules).

        """
        mut_res_lines = ''
        rpk_res_lines = ''
        out_dir = 'run'
        input_pdb_filename = 'protein.pdb'

        # Create a copy of the pdb file
        pdb = self._pdb.copy()

        # Create the input file for Damietta
        spec_input_str = f"library\t{self._damietta_lib_path}\n"
        spec_input_str += f"input\t{input_pdb_filename}\n"
        spec_input_str += "\n"

        spec_input_str += f"out_dir\t{out_dir}\n"
        spec_input_str += "\n"

        for mutation in mutations:
            chainid, resid, new_resname = mutation.split(':')
            residue = f'{chainid}:{resid}'
            try:
                mut_res_lines += f"mut_res\t{self._residue_to_indices[residue]}\t{new_resname}\n"
            except KeyError:
                raise RuntimeError(f'Residue {residue} not found in the pdb file.')

        spec_input_str += mut_res_lines
        spec_input_str += "\n"

        if repack_neighbor_residues:
            selection_str = []
            residues_to_not_repack = []

            for mutation in mutations:
                chainid, resid, new_resname = mutation.split(':')
                residue = f'{chainid}:{resid}'
                selection_str.append(f'(resid {self._residue_to_indices[residue]})')
                residues_to_not_repack.append(self._residue_to_indices[residue])

            selection_str = ' or '.join(selection_str)
            print(selection_str)

            residues = pdb.select(selection_str).copy()
            contacts = pdb.select(f'calpha and (same residue as within {neighbor_cutoff} of residues)', residues=residues)

            for atom in contacts:
                atom_index = atom.getResindex()

                if not atom_index in residues_to_not_repack:
                    rpk_res_lines += f"rpk_res\t{atom_index}\n"
        else:
            # We need to at least add the mutated residues for repacking
            for mutation in mutations:
                chainid, resid, new_resname = mutation.split(':')
                residue = f'{chainid}:{resid}'
                rpk_res_lines += f"rpk_res\t{self._residue_to_indices[residue]}\n"

        spec_input_str += rpk_res_lines
        spec_input_str += "\n"

        spec_input_str += f"scramble_order\t1\n"
        spec_input_str += "\n"

        if load_memory:
            spec_input_str += f"load_memory\t1\n"
        else:
            spec_input_str += f"load_memory\t0\n"

        # Run Damietta
        with temporary_directory(prefix='dm_mut_', dir='.', clean=clean) as tmp_dir:
            # Renumbers the residues in the pdb file, starting from 1
            residue_indices = pdb.getResindices()
            pdb.setResnums(residue_indices + 1)

            # Writes the pdb file
            writePDB(input_pdb_filename, pdb, renumber=False)

            # Modify PDB file, because Damietta is a special kid
            format_pdb_for_damietta(input_pdb_filename, input_pdb_filename)

            with open(f'input.spec', 'w') as spec_input_file:
                spec_input_file.write(spec_input_str)

            run_sinai_cs_f2m2f(f'input.spec', self._damietta_path)

            # Read the output pdb file
            with open(f'{out_dir}/init.pdb', 'r') as f:
                while True:
                    line = f.readline()

                    if line.startswith('REMARK AVERAGE ENERGY PER RESIDUE:'):
                        sl = line.split()
                        energies = np.array([float(sl[6]), float(sl[8]), float(sl[10]), float(sl[12]), float(sl[14]), float(sl[16])])
                        break

            # Replace the pdb with the mutated one
            self._pdb = parsePDB('run/init.pdb')

        return energies

    def export_pdb(self, output_pdb_filename):
        """
        Writes the pdb file.

        Parameters
        ----------
        output_pdb_filename : str
            The filename of the output pdb file.

        """
        writePDB(output_pdb_filename, self._pdb)
