#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MHC-I
#

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from MDAnalysis.analysis.rms import rmsd
from scipy.stats import pearsonr

from helm import build_helm_string

def read_pssm_file(pssm_file):
    data = []
    AA = []

    with open(pssm_file) as f:
        lines = f.readlines()
        
        n_columns = int(lines[0].split('\t')[1])

        for line in lines[1:-1]:
            sline = line.strip().split('\t')
            AA.append(sline[0])
            data.append([float(v) for v in sline[1:]])

    columns = list(range(1, n_columns + 1))
    pssm = pd.DataFrame(data=data, columns=columns, index=AA)
    
    return pssm


class MHCIPeptideScorer:
    
    def __init__(self, pssm_files, mhci_dataset, energy_cutoff=-5.0):
        self._pssm = {}
        self._reg = {}
        self._ref_global = None
        self._energy_cutoff = energy_cutoff
        
        # Read PSS matrices
        for pssm_file in pssm_files:
            pssm = read_pssm_file(pssm_file)
            self._pssm[len(pssm.columns)] = pssm
        
        # Score peptides using those PSSM
        pssm_scores = self.score(mhci_dataset['sequence'])
        
        # Fit PSSM scores to experimental values
        reg = LinearRegression()
        reg.fit(pssm_scores[:, None], mhci_dataset.energy)
        print('----- Peptide global -----')
        print('N peptide: %d' % mhci_dataset.shape[0])
        print('R2: %.3f' % reg.score(pssm_scores[:, None], mhci_dataset.energy))
        print('RMSD : %.3f kcal/mol' % rmsd(reg.predict(pssm_scores[:, None]), mhci_dataset.energy))
        print('')
        self._reg = reg 
            
    def score(self, sequences):
        # Score peptides using those PSSM
        scores = []
                                 
        for sequence in sequences:
            score = 0
            
            try:
                pssm = self._pssm[len(sequence)]
            except:
                # We cannot score that peptide, so default score is 999
                score = 999
                scores.append(score)
                continue
                
            for i, aa in enumerate(sequence):
                score += pssm.loc[aa][i + 1]

            scores.append(score)

        scores = np.array(scores)
    
        return scores
            
    def predict_energy(self, sequences):
        scores = []
        
        pssm_scores = self.score(sequences)
        pred_scores = self._reg.predict(pssm_scores[:, None])
        
        # Apply cutoff condition
        pred_scores[pred_scores > self._energy_cutoff] = 0.

        return pred_scores
    
    def generate_random_peptides(self, n_peptides, energy_bounds, peptide_lengths, return_predicted_energy=True):
        random_peptides = []
        random_peptide_scores = []
        
        if not isinstance(peptide_lengths, (list, tuple)):
            peptide_lengths = [peptide_lengths]
        
        # We don't care about which pssm we are using here
        keys = list(self._pssm.keys())
        AA = self._pssm[keys[0]].index

        while True:
            peptide_length = np.random.choice(peptide_lengths)
            
            p = ''.join(np.random.choice(AA, peptide_length))
            
            if return_predicted_energy:
                s = self.predict_energy([p])[0]
            else:
                s = self.score([p])[0]

            if energy_bounds[0] <= s <= energy_bounds[1]:
                helm_string = build_helm_string({'PEPTIDE1': p}, [])

                random_peptides.append(helm_string)
                random_peptide_scores.append(s)
                #print(len(random_peptides), helm_string)

            if len(random_peptides) == n_peptides:
                break
        
        sorted_index = np.argsort(random_peptide_scores)
        random_peptides = np.array(random_peptides)[sorted_index]
        random_peptide_scores = np.array(random_peptide_scores)[sorted_index]

        return random_peptides, random_peptide_scores
