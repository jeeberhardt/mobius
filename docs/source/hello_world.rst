.. _hello_world:

Hello, world!
=============

This example shows the basic usage of Mobius, and how to use Bayesian Optimisation (BO) to 
optimize peptides against the MHC class I HLA-A*02:01. Starting from a single peptide sequence
with a low binding affinity, the goal is to improve the binding afffinity in only few DMT 
iterations.

First, import the required modules and functions from the Mobius package, as well as NumPy:

.. code-block:: python

    import numpy as np

    from mobius import Planner, SequenceGA
    from mobius import Map4Fingerprint
    from mobius import GPModel, ExpectedImprovement, TanimotoSimilarityKernel
    from mobius import LinearPeptideEmulator
    from mobius import homolog_scanning, alanine_scanning
    from mobius import convert_FASTA_to_HELM

Now, create a simple linear peptide emulator for MHC class I HLA-A*02:01.

.. note::
    This is for benchmarking or demonstration purposes only and should be replaced 
    with actual lab experiments. See list of other available emulators in the 
    :ref:`mobius` section.

.. note::
    The Position Specific Scoring Matrix (PSSM) files used to initialize the emulator are
    available in the `IEDB <http://tools.iedb.org/mhci/download/>`_ database. Click on the 
    following link to directly download the PSSM files: `IEDB_MHC_I-2.9_matx_smm_smmpmbec.zip <http://tools.immuneepitope.org/static/download/IEDB_MHC_I-2.9_matx_smm_smmpmbec.tar.gz>`_.

.. code-block:: python

    pssm_files = ['data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-8.txt',
                  'data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-9.txt',
                  'data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-10.txt',
                  'data/mhc/IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-02:01-11.txt']
    lpe = LinearPeptideEmulator(pssm_files)

Define the peptide sequence to optimize, and convert it to HELM format. Mobius uses internally
the HELM notation to encode peptide sequences. The HELM format is a standard format for 
representing peptides and other complex biomolecules. See documentation from the 
`Pistollia Alliance <https://www.pistoiaalliance.org/helm-notation/>`_ to know more about the
HELM notation.

.. warning::
    Mobius can only handle peptides or equivalent polymers. RNA and oligonucleotides are not supported.

.. code-block:: python

    lead_peptide = convert_FASTA_to_HELM('HMTEVVRRC')[0]

Generate the initial seed library of 96 peptides using a combination of 
alanine scanning and homolog scanning sequence-based strategies:

.. note::
    Other peptide scanning strategies are available. See list of the other available
    scanning strategies in the :ref:`mobius` section.

.. code-block:: python

    seed_library = [lead_peptide]

    for seq in alanine_scanning(lead_peptide):
        seed_library.append(seq)
        
    for seq in homolog_scanning(lead_peptide):
        seed_library.append(seq)

        if len(seed_library) >= 96:
            print('Reach max. number of peptides allowed.')
            break

Virtually test the seed library using the linear peptide emulator. It outputs
the pIC50 value for each tested peptides. A pic50 value of 0 correspond to 
an IC50 of 1 nM, a pIC50 of 1 corresponds to an IC50 of 10 nM, etc.

.. note::
    This is for benchmarking or demonstration purposes only and should be replaced 
    with actual lab experiments.

.. code-block:: python

    pic50_seed_library = lpe.predict(seed_library)

Now that we have results from the initial lab experiment, we can start the Bayesian 
Optimization. Define the molecular fingerprint, the surrogate model (Gaussian Process), 
the acquisition function (Expected Improvement) and the optimization method (SequenceGA):

.. note::
    Other molecular fingerprints, surrogate models and acquisitions functions are available. 
    See list of the other available molecular fingerprints and surrogate models in the 
    :ref:`mobius` section.

.. code-block:: python

    map4 = Map4Fingerprint(input_type='helm_rdkit', dimensions=4096, radius=1)
    gpmodel = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
    acq = ExpectedImprovement(gpmodel, maximize=False)
    optimizer = SequenceGA(total_attempts=5)

Define the search protocol in a YAML configuration file (`design_protocol.yaml`) that will be used 
to optimize peptide sequences using the acquisition function. See the :ref:`design_protocol` section
for more details about the design protocol. This YAML configuration file defines the design
protocol, which includes the peptide scaffold, linear here, and sets of monomers for some positions to be used
during the optimization. Finally, it defines the optimizer, here SequenceGA, to optimize the peptide sequences
using the acquisition function / surrogate model initialized earlier.

.. code-block:: YAML

    design:
      monomers: 
        default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
        APOLAR: [A, F, G, I, L, P, V, W]
        POLAR: [C, D, E, H, K, N, Q, R, K, S, T, M]
        AROMATIC: [F, H, W, Y]
        POS_CHARGED: [K, R]
        NEG_CHARGED: [D, E]
      scaffolds:
        - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0:
            PEPTIDE1:
              1: [AROMATIC, NEG_CHARGED]
              4: POLAR
              9: [A, V, I, L, M, T]
    filters:
      - class_path: mobius.PeptideSelfAggregationFilter
      - class_path: mobius.PeptideSolubilityFilter
        init_args:
          hydrophobe_ratio: 0.5
          charged_per_amino_acids: 5

And instantiate the Planner object using the YAML configuration file, the acquisition function
and the optimization method:

.. code-block:: python

    ps = Planner(acq, optimizer, design_protocol='design_protocol.yaml')

Run three Design-Make-Test cycles, iterating through the following steps:

#. Recommend 96 new peptides based on existing data using the Bayesian optimization.
#. Optionally, apply additional filtering methods to the suggested peptides.
#. Virtually test the suggested peptides using the MHC emulator (replace with actual lab experiments).
#. Update the list of tested peptides and their pIC50 values.

.. code-block:: python

    peptides = list(seed_library)[:]
    pic50_scores = list(pic50_seed_library)[:]

    for i in range(3):
        suggested_peptides, _ = ps.recommand(peptides, pic50_scores, batch_size=96)

        # Here you can add whatever methods you want to further filter out peptides

        # Virtually test the suggested peptides using the MHC emulator
        # You know the drill now, this is for benchmarking or demonstration 
        # purposes only and should be replaced with actual lab experiments.
        pic50_suggested_peptides = lpe.predict(suggested_peptides)
        
        peptides.extend(list(suggested_peptides))
        pic50_scores.extend(list(pic50_suggested_peptides))
        
        best_seq = peptides[np.argmin(pic50_scores)]
        best_pic50 = np.min(pic50_scores)
        print('Best peptide found so far: %s / %.3f' % (best_seq, best_pic50))
        print('')

By the end of the optimization loop, the best peptide sequence and its pIC50 score 
will be printed. This tutorial demonstrates how to use Bayesian optimization for 
peptide sequence optimization in a Design-Make-Test closed-loop platform. Remember 
to replace the emulator steps with actual lab experiments in a real-world application.
