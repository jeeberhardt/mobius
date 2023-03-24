#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius
#

import numpy as np
import pandas as pd


class Mobius:
    """
    Class for benchmarking sampling protocol against one emulator/oracle.

    """

    def __init__(self):
        pass
    
    def run(self, polymers, values, emulator, sampler, num_iter=5, batch_size=96):
        """
        Function for running the benchmark optimization process for polymers/peptides 
        against the `Emulator` (oracle).

        Parameters
        ----------
            polymers : list of str
                Polymers/peptides in HELM format.
            values : list of int or float
                Values associated to each polymer/peptide.
            emulator : `Emulator`
                Emulator (oracle) used to simulate actual lab experiments.
            sampler : Sampler`
                Protocol used for sampling the polymer/peptide sequence space.
            num_iter : int, default: 5
                Total number of optimization cycles.
            batch_size : int, default: 96
                Size of the batches, number of polymers/peptides returned after 
                each optimization cycle

        Returns
        -------
        results : pd.DataFrame
            Pandas DataFrame containing all the results from the optimization 
            process with the following columns: ['iter', 'polymer', 'exp_value', 
            'pred_value']

        """
        data = []

        all_suggested_polymers = np.asarray(polymers).copy()
        all_exp_values = np.asarray(values).copy()

        # Add initial population
        for p, ev in zip(all_suggested_polymers, all_exp_values):
            data.append((0, p, ev, np.nan))

        for i in range(num_iter):
            suggested_polymers, predicted_values = sampler.recommand(all_suggested_polymers, all_exp_values, batch_size)

            exp_values = emulator.predict(suggested_polymers)

            all_suggested_polymers = np.concatenate([all_suggested_polymers, suggested_polymers])
            all_exp_values = np.concatenate([all_exp_values, exp_values])

            for p, ev, pv in zip(suggested_polymers, exp_values, predicted_values):
                data.append((i + 1, p, ev, pv))

        columns = ['iter', 'polymer', 'exp_value', 'pred_value']
        df = pd.DataFrame(data=data, columns=columns)

        return df

    def benchmark(self, polymers, values, emulator, samplers, num_iter=5, batch_size=96, num_independent_run=5):
        """
        Function to benchmark multiple sampling strategies for polymers/peptide 
        optimization against the `Emulator` (oracle).

        Parameters
        ----------
            polymers : list of str
                Polymers/peptides in HELM format.
            values : list of int or float
                Values associated to each polymer/peptide.
            emulator : `Emulator`
                Emulator (oracle) used to simulate actual lab experiments.
            samplers : dict of `Sampler`
                Dictionary of protocols used for sampling the polymer/peptide 
                sequence space.
            num_iter : int, default: 5
                Total number of optimization cycles.
            batch_size : int, default: 96
                Size of the batches, number of polymers/peptides returned after 
                each optimization cycle.
            num_independent_run :, int, default: 5
                Total number of independent runs to execute for each sampling 
                protocol.

        Returns
        -------
        results : pd.DataFrame
            Pandas DataFrame containing all the results from the optimization process with
            the following columns: ['sampler', 'run', 'iter', 'polymer', 'exp_value', 'pred_value']

        """
        dfs = []

        assert isinstance(samplers, dict), 'Samplers must be defined as a dictionary ({\'sampler1_name\': sampler})'

        for sampler_name, sampler in samplers.items():
            for i in range(num_independent_run):
                df = self.run(polymers, values, emulator, sampler, num_iter, batch_size)

                df['run'] = i + 1
                df['sampler'] = sampler_name
                df = df[['sampler', 'run', 'iter', 'polymer', 'exp_value', 'pred_value']]

                dfs.append(df)

        dfs = pd.concat(dfs)

        return dfs
