#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - virtual target
#

import numpy as np
import pandas as pd


class VirtualTarget:
	"""Class to handle a virtual target"""
	def __init__(self, forcefield, seed=None):
		"""Initialize the virtual target


		Args:
			forcefield (Forcefield): a forcefield to score interaction between
				a sequence and the virtual target
			seed (int): random seed

		"""
		self._forcefield = forcefield
		self._sequence_length = None
		self._random_seed = seed
		self._rng = np.random.default_rng(self._random_seed)

	def __repr__(self):
		repr_str = ""
		return repr_str

	def load_pocket(input_filename):
		"""Load virtual target
		
		Args:
			input_filename (str): input csv filename containing the virtual
				target

		"""
		self._pocket = pd.read_csv(input_filename)

	def generate_random_pocket(self, sequence_length=None):
		"""Generate a random pocket

		Args:
			sequence_length (int): length of the sequence to generate

		"""
		columns = ['solvant_exposed', 'hb_type', 'charge', 'length', 'volume']
		self._sequence_length = sequence_length

		for i in range(self._sequence_length):
			solvant_exposed = self._rng.choice([0, 1])

			if solvant_exposed:
				hb_type = self._rng.choice(['D', 'A', 'DA'])
			else:
				hb_type = self._rng.choice(['H',  'D', 'A', 'DA'])

			charge = self._rng.choice(['H', 'D', 'A', 'DA'])
			length = self._rng.random()
			volume = self._rng.random()

			data.append([solvant_exposed, hb_type, charge, length, volume])
		
		pocket = pd.DataFrame(data=data, columns=columns)

		self._pocket = pocket

	def score_peptides(peptides):
		"""Score interaction between peptides and the virtual target 
		using the provided forcefield

		Args:
			peptides (list): list of peptide strings

		Returns:
			np.ndarray: array of score for each peptide

		"""
		score = []

		for p in peptides:
			score.append(self._forcefield.score(p))

		return np.array(score) 

	def export_pocket(output_filename):
		"""Export virtual target as json file

		Args:
			output_filename (str): output csv filename

		"""
		self._pocket.to_csv(output_filename, index=False)
