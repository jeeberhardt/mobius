.. _moo:

Multi-objectives optimization
=============================

So far, we only had considered single-objective optimization problems. However,
many real-world problems involve multiple objectives. For example, in our context 
we may want to optimize peptides that bind to multiple MHC class I receptors, at 
the same time, or only one and not the other, or none of them (why not?). In this 
section, we will show how to do multi-objectives optimization.

For this example, we take two different MHC class I receptors, HLA-A*32:15 and
HLA-A*26:02, and try to optimize peptides that bind to both receptors. As for the
:ref:`single-objective optimization tutorial <hello_world>`, we first need to import all
the necessary modules.

.. code-block:: python

    import numpy as np

    from mobius import Planner, SequenceGA
    from mobius import Map4Fingerprint
    from mobius import GPModel, ExpectedImprovement, TanimotoSimilarityKernel
    from mobius import LinearPeptideEmulator
    from mobius import homolog_scanning, alanine_scanning
    from mobius import convert_FASTA_to_HELM

Now, create two linear peptide emulators for MHC class I HLA-A*32:15 and HLA-B*26:02.

.. note::
    The Position Specific Scoring Matrix (PSSM) files used to initialize the emulator are
    available in the `IEDB <http://tools.iedb.org/mhci/download/>`_ database. Click on the 
    following link to directly download the PSSM files: `IEDB_MHC_I-2.9_matx_smm_smmpmbec.zip <http://tools.immuneepitope.org/static/download/IEDB_MHC_I-2.9_matx_smm_smmpmbec.tar.gz>`_.

.. code-block:: python

    # Create the linear peptide emulators
    hla_a32 = LinearPeptideEmulator('IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-32:15-9.txt')
    hla_a26 = LinearPeptideEmulator('IEDB_MHC_I-2.9_matx_smm_smmpmbec/smmpmbec_matrix/HLA-A-26:02-9.txt')

Again, we generate the initial seed library of 96 peptides, starting with one single lead 
peptide, using a combination of alanine scanning and homolog scanning sequence-based 
strategies:

.. code-block:: python

    # Predicted to bind to both HLA-A*32:15 and HLA-A*26:02 with
    # pIC50 of 3.00 and 3.77, respectively.
    lead_peptide = 'PEPTIDE1{Q.F.V.R.T.N.D.I.I}$$$$V2.0'
    seed_library = [lead_peptide]

    for seq in alanine_scanning(lead_peptide):
        seed_library.append(seq)
        
    for seq in homolog_scanning(lead_peptide):
        seed_library.append(seq)

        if len(seed_library) >= 96:
            print('Reach max. number of peptides allowed.')
            break

    # Virtually test the seed library using the MHC emulators
    hla_a32_scores = hla_a32.score(seed_library)
    hla_a26_scores = hla_a26.score(seed_library)
    pic50_seed_library = np.column_stack((hla_a32_scores, hla_a26_scores))

Now that we have results from the initial lab experiment, we can start the Bayesian 
Optimization. Now, instead of using a single Gaussian Process model, we will use two
models, one for each MHC class I receptor. The rest stays the same. Well, almost! We 
just need to change the classical `GA` for the `SMS-EMOA` algorithm. For more 
information about the `SMS-EMOA` algorithm, please refer to the 
`pymoo documentation <https://pymoo.org/algorithms/moo/sms.html>`_.

.. code-block:: python

    map4 = Map4Fingerprint(input_type='helm', dimensions=4096, radius=1)

    # Create the Gaussian Process models
    gpmodel_a32 = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
    gpmodel_a26 = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=map4)
    
    # .. pass them to the acquisition function
    # Here we want to both minimize the pIC50 values for HLA-A*32:15 and HLA-A*26:02
    # so we set the `maximize` argument to `False` for both models. But you can also
    # set it to `True` for one of the models, and `False` for the other one, depending
    # on your optimization problem.
    acq = ExpectedImprovement([gpmodel_a32, gpmodel_a26], maximize=[False, False])

    # Instead of using the classic GA, we are going to use the SMS-EMOA algorithm
    optimizer = SequenceGA(algorithm='SMSMOEA', period=15)

    ps = Planner(acq, optimizer)

    peptides = seed_library.copy()
    pic50_scores = pic50_seed_library.copy()

    for i in range(5):
        suggested_peptides, _ = ps.recommand(peptides, pic50_scores, batch_size=96)

        # Virtually test the suggested peptides using the MHC emulators
        # This is for benchmarking or demonstration purposes only and 
        # should be replaced with actual lab experiments.
        exp_values = []
        for emulator in [hla_a32, hla_a26]:
            values = np.asarray(emulator.score(suggested_polymers))
            exp_values.append(values)
        exp_values = np.stack(exp_values, axis=1)

        # Add the suggested peptides to the library, and start over
        peptides = np.concatenate([apeptides, suggested_polymers])
        pic50_scores = np.vstack([pic50_scores, exp_values])

.. note::

    The batch of peptides suggested by the planner is not necessaryly optimal for
    your particular problem, and you might want to explore different way of selecting
    peptides from the planner's output. For example, you might want to do some 
    clustering and select only the peptides that are the most representative of 
    each cluster.

    Well that's easy, the raw results (from the last GA generation, and just before 
    the batch selection) are stored in the `planner` object. You can access them using
    the `_results` attribute at the end of the GA optimization:

    .. code-block:: python

        # Get the raw results from the GA sampling
        raw_peptides, raw_scores = planner._results
