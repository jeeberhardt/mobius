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


class DamiettaScorer:

    def __init__(self, pdb_filename, damietta_path):
        """
        Loads a pose from filename with the params in the params_folder

        Parameters
        ----------
        filename : str
            The filename of the pdb file, prepared for Damietta.
        damietta_path : str
            The path of the Damietta directory.

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
        self._residue_to_indices = {f'{r.getChid()}:{r.getResnum()}': i + 1 for i, r in enumerate(self._pdb.iterResidues())}

    def mutate_and_score(self, mutations, repack_neighbor_residues=False, neighbor_cutoff=4.0, load_memory=True, clean=True):
        """
        Mutates the protein/peptide with the given mutations list.

        Parameters
        ----------
        mutations : list of str
            The list of mutations in the format chainid:resid:resname.
        repack_neighbor_residues : bool, optional
            Whether to repack the residues within the given cutoff distance of the mutated residues.
            Default is False.
        neighbor_cutoff : float, optional
            The cutoff distance to consider the neighboring residues. Default is 4.0 Angstrom.
        load_memory : bool, optional
            Whether to load the memory. Default is True.
        clean : bool, optional
            Whether to clean the temporary files. Default is True.

        Returns
        -------
        energies : ndarray of float
            The energies (in kcal/mol) of the mutated protein. The energies returned
            are in the following order: (pp_dG, k_dG, lj, solv, elec, total).

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
        with temporary_directory(prefix='dm_', dir='.', clean=clean) as tmp_dir:
            # Renumbers the residues in the pdb file, starting from 1
            residue_indices = pdb.getResindices()
            pdb.setResnums(residue_indices + 1)

            # Writes the pdb file
            writePDB(input_pdb_filename, pdb, renumber=False)

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

        return energies

