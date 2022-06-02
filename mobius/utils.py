#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - utils
#

from importlib import util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def path_module(module_name):
    specs = util.find_spec(module_name)
    if specs is not None:
        return specs.submodule_search_locations[0]
    return None


def function_equal(func_1, func_2):
    return func_1.__code__.co_code == func_2.__code__.co_code


def opposite_signs(x, y):
    return ((x ^ y) < 0)


def affinity_binding_to_energy(value, unit='nM', temperature=300.):
    unit_converter = {'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1}
    RT = 0.001987 * temperature
    return RT * np.log(value * unit_converter[unit])


def energy_to_affinity_binding(value, unit='nM', temperature=300.):
    unit_converter = {'pM': 1e-12, 'nM': 1e9, 'uM': 1e6, 'mM': 1e3, 'M': 1}
    RT = 0.001987 * temperature
    return np.exp(value / RT) * unit_converter[unit]


def plot_results(df, run_name):
    fig, axarr = plt.subplots(1, 4, sharex=True, figsize=(25, 7.5))

    for i in range(2):
        x = [-1, 0, 1, 2, 3, 4, 5]
        axarr[i].plot(x, [affinity_binding_to_energy(1, unit='mM')] * len(x), '--', linewidth=1, color='lightgray')
        axarr[i].plot(x, [affinity_binding_to_energy(1, unit='uM')] * len(x), '--', linewidth=1, color='gray')
        axarr[i].plot(x, [affinity_binding_to_energy(1, unit='nM')] * len(x), '--', linewidth=1, color='black')
        axarr[i].text(-0.4, affinity_binding_to_energy(1, unit='mM'), 'mM')
        axarr[i].text(-0.4, affinity_binding_to_energy(1, unit='uM'), 'uM')
        axarr[i].text(-0.4, affinity_binding_to_energy(1, unit='nM'), 'nM')

    # Average
    sns.violinplot(x='gen', y='exp_score', data=df[df['exp_score'] != 0.], bw=.2, scale='width', cut=1, linewidth=0.5, color="0.9", ax=axarr[0])
    sns.boxplot(x='gen', y='exp_score', data=df[df['exp_score'] != 0.], width=.2, boxprops={'zorder': 2}, color='tab:blue', ax=axarr[0])

    # Min
    mins = df.loc[df.groupby(by=['sample', 'gen'])['exp_score'].idxmin()][['gen', 'exp_score']]
    sns.violinplot(x='gen', y='exp_score', data=mins, bw=.2, scale='width', cut=1, linewidth=0.5, color="0.9", ax=axarr[1])
    sns.swarmplot(x='gen', y='exp_score', data=mins, color='tab:blue', size=7, alpha=0.8, ax=axarr[1])

    # Length
    y = df.groupby(by=['sample', 'gen', 'length'])\
      .agg(['count'])['exp_score']\
      .groupby(['gen', 'length'])\
      .mean()\
      .fillna(0)\
      .pivot_table(['count'], ['gen'], 'length')\
      .reset_index()\
      .fillna(0)

    # fillna(0) to avoid NaN otherwise removed when pivoting...
    errors = df.groupby(by=['sample', 'gen', 'length'])\
      .agg(['count'])['exp_score']\
      .groupby(['gen', 'length'])\
      .std()\
      .fillna(0)\
      .pivot_table(['count'], ['gen'], 'length')\
      .reset_index()\
      .fillna(0)

    y.plot.bar(x='gen', yerr=errors, ax=axarr[2], capsize=4, rot=0, fontsize=10, stacked=True)

    # Binders and non binders
    data_mean = []
    data_errors = []

    for gen, gen_group in df.groupby(by='gen'):
        n_sample = gen_group['sample'].unique().shape[0]

        mean_binders = gen_group[gen_group['exp_score'] != 0].shape[0] / n_sample
        mean_nonbinders = gen_group[gen_group['exp_score'] == 0].shape[0] / n_sample

        tmp_binders = []
        tmp_nonbinders = []
        for sample, sample_group in gen_group.groupby(by='sample'):
            tmp_binders.append(sample_group[sample_group['exp_score'] != 0].shape[0])
            tmp_nonbinders.append(sample_group[sample_group['exp_score'] == 0].shape[0])

        std_binders = np.std(tmp_binders)
        std_nonbinders = np.std(tmp_nonbinders)

        data_mean.append((gen, mean_binders, mean_nonbinders))
        data_errors.append((gen, std_binders, std_nonbinders))

    df_mean_binders = pd.DataFrame(data=data_mean, columns=('gen', 'binders', 'non_binders'))
    df_errors_binders = pd.DataFrame(data=data_errors, columns=('gen', 'binders', 'non_binders'))
    df_mean_binders.plot.bar(x='gen', y=['binders', 'non_binders'], yerr=df_errors_binders, ax=axarr[3], capsize=4, rot=0, fontsize=10, stacked=True)


    axarr[0].set_xlabel('Generations', fontsize=20)
    axarr[1].set_xlabel('Generations', fontsize=20)
    axarr[2].set_xlabel('Generations', fontsize=20)
    axarr[3].set_xlabel('Generations', fontsize=20)
    axarr[0].set_ylabel('Exp. binding (kcal/mol)', fontsize=20)
    axarr[2].set_ylabel('#Peptides', fontsize=20)

    axarr[0].set_ylim([-16, 0])
    axarr[1].set_ylim([-16, 0])
    axarr[2].set_ylim([0, 160])
    axarr[3].set_ylim([0, 160])

    #axarr[0].legend(['Mean'], fontsize=15)
    #axarr[1].legend(['Min'], fontsize=15)
    axarr[2].legend(['%d-mer' % s for s in np.sort(df['length'].unique())], fontsize=10)
    axarr[3].legend(['Binders', 'Non binders'], fontsize=10)
    
    plt.savefig('figure_%s.png' % run_name, dpi=300, bbox_inches='tight')
    plt.show()
