#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius
#

import numpy as np
import pandas as pd


class Mobius:

    def __init__(self):
        pass
    
    def run(self, polymers, values, emulator, sampler, num_iter=5, batch_size=96):
        """
        Run function to optimize polymers

        Arguments:
        ----------
            polymers: List of polymers in HELM format
            values: List of values associated to each polymer
            emulator: Emulator
            sampler: Sampler
            num_iter: number of iterations (default: 5)
            batch_size: size of the batches (default: 96)

        """
        data = []

        all_suggested_polymers = np.asarray(polymers).copy()
        all_exp_values = np.asarray(values).copy()

        # Add initial population
        for p, ev in zip(all_suggested_polymers, all_exp_values):
            data.append((0, p, ev, np.nan))

        for i in range(num_iter):
            suggested_polymers, predicted_values = sampler.recommand(all_suggested_polymers, all_exp_values, batch_size)

            suggested_polymers_fasta = [''.join(c.split('$')[0].split('{')[1].split('}')[0].split('.')) for c in suggested_polymers]
            exp_values = emulator.predict(suggested_polymers_fasta)

            all_suggested_polymers = np.concatenate([all_suggested_polymers, suggested_polymers])
            all_exp_values = np.concatenate([all_exp_values, exp_values])

            for p, ev, pv in zip(suggested_polymers, exp_values, predicted_values):
                data.append((i + 1, p, ev, pv))

        columns = ['iter', 'polymer', 'exp_value', 'pred_value']
        df = pd.DataFrame(data=data, columns=columns)

        return df

    def benchmark(self, polymers, values, emulator, samplers, num_iter=5, batch_size=96, num_independent_run=5):
        """
        Run function to benchmark sampler strategies

        Arguments:
        ----------
            polymers: List of polymers in HELM format
            values: List of values associated to each polymer
            emulator: Emulator
            sampler: Samplers
            num_iter: number of iterations (default: 5)
            batch_size: size of the batches (default: 96)
            num_independent_run: number of independent runs (default: 5)

        """
        dfs = []

        assert isinstance(samplers, dict), 'Samplers must be defined as a dictionary ({\'sampler1_name\': sampler})'

        for sampler_name, sampler in samplers.items():
            for i in range(num_independent_run):
                df = self.run(polymers, values, emulator, sampler, num_iter, batch_size)

                df['ind_run'] = i + 1
                df['sampler'] = sampler_name
                df = df[['sampler', 'ind_run', 'iter', 'polymer', 'exp_value', 'pred_value']]

                dfs.append(df)

        dfs = pd.concat(dfs)

        return dfs
