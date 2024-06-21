.. _non_standard:

Macrocycles and non-standard monomers
=====================================

Optimization of a macrocyclic peptide with non-standard monomers
----------------------------------------------------------------

Are you weary of those flashy, `Selling Sunset` methods that promise the world 
but fall short, functioning solely on linear peptides with the same old 20 
standard-amino acids? I feel you. That's exactly why `mobius` was created.
It natively handles complex peptide scaffolds, embracing the world of non-natural 
amino acids. 

To show you some of `mobius` capabilities we're going to kick things off with a 
simple (albeit not realistic) example. For this exercice, we are going to 
take a random `macrocyclic peptide (ID: 7170) <http://cycpeptmpdb.com/peptides/id_7170/>`_ 
containing non-natural amino acids from the `CycPeptMPDB <http://cycpeptmpdb.com/>`_, 
and try to find it back starting from a peptide containing only standard amino acids.

.. code-block:: python

    import numpy as np

    from mobius import Planner, SequenceGA, FindMe
    from mobius import MHFingerprint
    from mobius import GPModel, ExpectedImprovement, TanimotoSimilarityKernel
    from mobius import homolog_scanning, alanine_scanning

    # This is the peptide sequence we need to find
    # See http://cycpeptmpdb.com/peptides/id_7170/ for more information
    target = 'PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0'

In our quest to rediscover the target sequence from just standard amino acids, we'll 
have to keep in mind that while mobius is able to do a lot of things, but it doesn't possess 
magical abilities (at least not yet!). So, we're proceeding with a couple of assumptions:

#. We're already informed about the peptide scaffold (10-mer, link between monomers 4 and 10).
#. We know the non-standard amino acids that need to be integrated into the mix.

.. note::

    While `mobius` does not do scaffold hopping on its own, you could define multiple 
    scaffolds to be explored during the optimization (see :ref:`design_protocol`).

To simulate a closed-loop Design-Make-Test optimization cycle, we're going to use the 
FindMe emulator. This clever tool works by setting up a target sequence—in this case, 
the sequence we defined earlier—and a kernel function, specifically the Tanimoto kernel, 
to evaluate our distance from the target. The end goal? To unearth the target sequence.

Since we're employing the Tanimoto metric, scores will range from 0 (indicating a 
completely different sequence) to 1 (indicating an identical sequence).

.. code-block:: python

    # Here we use the (folded) MinHashed Fingerprint method
    mhfp = MHFingerprint(input_type='helm', dimensions=4096, radius=3)
    kernel = TanimotoSimilarityKernel()

    # Here we define the FindMe emulator
    fm = FindMe(target, input_type='helm', kernel=kernel, transform=mhfp)

With the emulator now in place, our next step is to define our lead sequence, which 
only consists of standard amino acids. The Tanimoto score between the target 
and the lead sequence is about 0.394. Using this as our starting point, 
we'll generate a seed library comprising 96 sequences. These sequences are obtained 
by applying the homolog scanning method. To help the optimization process, we'll
also include the non-standard amino acids in the seed library.

.. code-block:: python

    lead = 'PEPTIDE1{E.P.L.T.A.K.I.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0'

    seed_library = [lead]

    for seq in homolog_scanning(lead):
    seed_library.append(seq)

    if len(seed_library) >= 48:
        print('Reach max. number of peptides allowed.')
        break

    for seq in monomers_scanning(lead, monomers=['Me_dL', 'd1-Nal', 'Nva', 'dL', 'Me_dA']):
        seed_library.append(seq)

        if len(seed_library) >= 96:
            print('Reach max. number of peptides allowed.')
            break
    
    # And we test right away the seed library using our emulator
    scores_seed_library = fm.score(seed_library)

Next, we set up our Gaussian Process Regression (GPR) model using the Tanimoto kernel 
function and the MinHashed fingerprint method, both defined earlier. For the acquisition 
function, we'll use the expected improvement with the ultimate aim of maximizing 
the Tanimoto score.

.. code-block:: python

    gpmodel = GPModel(kernel=TanimotoSimilarityKernel(), transform=mhfp)
    acq = ExpectedImprovement(gpmodel, maximize=True)

When it comes to the design protocol, we define all the non-standard amino acids that 
will be available during the optimization. We categorize them into two distinct monomer 
collections, aptly named `special` and `nter`. The N-terminal acetic acid (ac) is singled 
out since it can only be placed at the N-terminal part of the peptide. This gives us 
the flexibility for each position in the peptide chain to hold either a standard amino 
acid or one from the special collection. 

.. code-block:: yaml

    design:
      monomers: 
        default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
        special: [Me_dL, d1-Nal, Nva, dL, Me_dA]
        nter : [ac]
      polymers:
        - PEPTIDE1{X.X.X.T.X.X.X.X.X.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0:
            PEPTIDE1:
              1: [default, nter]
              2: [default, special]
              3: [default, special]
              5: [default, special]
              6: [default, special]
              7: [default, special]
              8: [default, special]
              9: [default, special]

With all the parameters now set up, we're ready to kickstart the optimization process! In 
this particular instance, we'll restrict ourselves to just 5 Design-Make-Test (DMT) cycles. 
So, without further ado, let's get this optimization rolling!

.. code-block:: python

    optimizer = SequenceGA(algorithm='GA', period=15, design_protocol_filename='sampling_macrocycle.yaml')
    ps = Planner(acq, optimizer)

    peptides = seed_library.copy()
    scores = scores_seed_library.copy()

    for i in range(5):
        suggested_peptides, _ = ps.recommend(peptides, scores.reshape(-1, 1), batch_size=96)

        # Here you can add whatever methods you want to further filter out peptides

        # Virtually test the suggested peptides using the MHC emulator
        # You know the drill now, this is for benchmarking or demonstration
        # purposes only and should be replaced with actual lab experiments.
        scores_suggested_peptides = fm.score(suggested_peptides)

        peptides = np.concatenate([peptides, suggested_peptides])
        scores = np.concatenate((scores, scores_suggested_peptides), axis=0)

        best_seq = peptides[np.argmax(scores)]
        best_scores = np.max(scores)
        print('Best peptide found so far: %s / %.3f' % (best_seq, best_scores))
        print('')

Typically, you'd see output similar to the following:

.. code-block:: none

    Reach max. number of peptides allowed.
    =================================================
    n_gen  |  n_eval  |     f_avg     |     f_min    
    =================================================
        1 |        0 |  9.994160E+02 |  9.993780E+02
        2 |      500 | -2.903180E-05 | -2.029716E-03
        3 |     1000 | -8.865342E-05 | -2.364596E-03
        ...
        23 |    11000 | -5.129112E-03 | -6.048419E-03
        24 |    11500 | -5.149737E-03 | -6.048419E-03
        25 |    12000 | -5.166888E-03 | -6.048419E-03
    Best peptide found so far: PEPTIDE1{K.P.[Me_dL].T.A.K.[Nva].[d1-Nal].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.681
    =================================================
    n_gen  |  n_eval  |     f_avg     |     f_min    
    =================================================
        1 |        0 |  9.995303E+02 |  9.993780E+02
        2 |      500 | -4.593440E-05 | -5.979990E-03
        3 |     1000 | -6.972748E-05 | -5.979990E-03
        ...
        18 |     8500 | -2.804339E-03 | -6.214865E-03
        19 |     9000 | -2.836284E-03 | -6.214865E-03
        20 |     9500 | -2.871679E-03 | -6.214865E-03
    Best peptide found so far: PEPTIDE1{K.P.[Me_dL].T.[d1-Nal].[Nva].[d1-Nal].L.P.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.693
    =================================================
    n_gen  |  n_eval  |     f_avg     |     f_min    
    =================================================
        1 |        0 |  9.995849E+02 |  9.993780E+02
        2 |      500 | -6.739627E-06 | -9.320378E-04
        3 |     1000 | -2.525135E-05 | -1.521608E-03
        ..
        20 |     9500 | -1.024600E-03 | -3.324028E-03
        21 |    10000 | -1.026064E-03 | -3.324028E-03
        22 |    10500 | -1.028012E-03 | -3.324028E-03
    Best peptide found so far: PEPTIDE1{K.P.[Me_dL].T.[d1-Nal].[Nva].[d1-Nal].[Me_dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.700
    =================================================
    n_gen  |  n_eval  |     f_avg     |     f_min    
    =================================================
        1 |        0 |  9.995928E+02 |  9.993780E+02
        2 |      500 | -5.364803E-06 | -5.612711E-04
        3 |     1000 | -1.867951E-05 | -1.238944E-03
        ..
        19 |     9000 | -3.378079E-04 | -2.658069E-03
        20 |     9500 | -3.397515E-04 | -2.658069E-03
        21 |    10000 | -3.405738E-04 | -2.658069E-03
    Best peptide found so far: PEPTIDE1{K.P.[Me_dL].T.[dL].[Me_dA].[Nva].[d1-Nal].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.796
    =================================================
    n_gen  |  n_eval  |     f_avg     |     f_min    
    =================================================
        1 |        0 |  9.996075E+02 |  9.993780E+02
        2 |      500 | -6.011269E-12 | -2.480689E-09
        3 |     1000 | -1.727572E-10 | -4.100772E-08
        ...
        25 |    12000 | -3.302355E-05 | -2.168542E-03
        26 |    12500 | -3.345462E-05 | -2.168542E-03
        27 |    13000 | -3.429750E-05 | -2.168542E-03
    Best peptide found so far: PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 1.000

As you can see, it nailed it in the last generation! However, it took 5 generations to
get there. This shows that there's still room for significant improvements. Let's consider 
this a successful starting point and a call to further optimize our approach.

.. warning::

    Before we wrap up, it's important to note that this experiment is a simplified scenario 
    and it might differ from real-world applications. In actual experiments, your results may 
    contain uncertainties, and more challenging still, you might not obtain a clear outcome for 
    every peptide tested.

Adding non-standard monomers
----------------------------

In the previous example, we used a pre-defined set of non-standard amino acids. But what if
you want to add your own? No problem! `mobius` allows you to add non-standard amino acids
on the fly. Let's see how this works.

For this example, we again take a random `macrocyclic peptide (ID: 48)  <http://cycpeptmpdb.com/peptides/id_48/>`_ 
from the `CycPeptMPDB <http://cycpeptmpdb.com/>`_ database. This peptide, with the following 
HELM string `PEPTIDE1{A.A.L.[meV].L.F.F.P.I.T.G.D.[-pip]}$PEPTIDE1,PEPTIDE1,1:R1-12:R3$$$V2.0`,
contains two non-standard amino acids not defined in `mobius`, namely the 
`C-terminal piperidine <http://cycpeptmpdb.com/monomers/-pip/>`_ (`-pip`) and the 
`N-methyl-L-valine <http://cycpeptmpdb.com/monomers/meV/>`_ (`meV`). To integrate these non-standard 
amino acids into our optimization, you will need the following information for each monomer:

* `MonomerID`: The monomer ID, as used in the HELM string.
* `MonomerSmiles`: A Chemaxon eXtended SMILES (CXSMILES), which is an extended version of SMILES that allows extra special features. In this case, we use a CXSMILES to define the attachment points on the monomer (`R1`, `R2`, etc, ..). For more information, see the `Chemaxon documentation <https://docs.chemaxon.com/display/docs/chemaxon-extended-smiles-and-smarts-cxsmiles-and-cxsmarts.md>`_
* `MonomerType`: The monomer type, which can be either `Backbone` or `Terminal`.
* `NaturalAnalog`: The natural analog of the monomer, althought this is not mandatory to provide.
* `MonomerName`: The full name of the monomer, also not mandatory.
* `Attachments`:

    * `AttachmentID`: The attachment ID.
    * `AttachmentLabel`: The attachment label, as defined in the CXSMILES.
    * `CapGroupName`: The cap group name.
    * `CapGroupSmiles`: The cap group as a CXSMILES string.

All these information need to de defined in a YAML file (`extra_non_standard.yaml`) as follows:

.. code-block:: yaml

    [
        {
            "MonomerID": "meV",
            "MonomerSmiles": "CC(C)[C@H](N(C)[*])C([*])=O |$;;;;;;_R1;;_R2;$|",
            "MonomerType": "Backbone",
            "PolymerType": "PEPTIDE",
            "NaturalAnalog": "V",
            "MonomerName": "N-methyl-L-valine",
            "Attachments": [{
                    "AttachmentID": "R1-H",
                    "AttachmentLabel": "R1",
                    "CapGroupName": "H",
                    "CapGroupSmiles": "[*][H] |$_R1;$|"
                },
                {
                    "AttachmentID": "R2-OH",
                    "AttachmentLabel": "R2",
                    "CapGroupName": "OH",
                    "CapGroupSmiles": "O[*] |$;_R2$|"
                }
            ]
        },
        {
            "MonomerID": "-pip",
            "MonomerSmiles": "[*]N1CCCCC1 |$_R1;;;;;;$|",
            "MonomerType": "Terminal",
            "PolymerType": "PEPTIDE",
            "NaturalAnalog": "X",
            "MonomerName": "C-Terminal piperidine",
            "Attachments": [{
                    "AttachmentID": "R1-H",
                    "AttachmentLabel": "R1",
                    "CapGroupName": "H",
                    "CapGroupSmiles": "[*][H] |$_R1;$|"
                }
            ]
        }
    ]

Once you have defined your YAML file, using it is as simple as that:

.. code-block:: python

    from mobius.utils import MolFromHELM
    from mobius import Map4Fingerprint

    # This is the peptide sequence containing non-standard amino acids
    # that are not yet defined in the library shipped with mobius.
    peptide = 'PEPTIDE1{A.A.L.[meV].L.F.F.P.I.T.G.D.[-pip]}$PEPTIDE1,PEPTIDE1,1:R1-12:R3$$$V2.0'

    # We can either directly get a RDKit molecule from the HELM string.
    mol = MolFromHELM(peptide, HELM_extra_library_filename='extra_non_standard.yaml')

    # .. or we can use the Map4Fingerprint method to get the fingerprint
    # of the peptide. This method will automatically load the extra
    # monomers from the YAML file. The map4 object can then be used during 
    # the optimization process as shown in the previous example.
    map4 = Map4Fingerprint(input_type='helm', HELM_extra_library_filename='extra_monomers.json')

    # The rest of the code stays the same as in the previous example.

.. note::

    The YAML file can also be used to redefine any standard or non-standard amino acids. For example, if 
    you want to add an extra attachment point to the tyrosine, you can do it by adding the following lines 
    to the YAML file:

    .. code-block:: yaml

        {
            "MonomerID": "Y",
            "MonomerSmiles": "[*]Oc1ccc([C@@H][C@@H](N[*])C([*])=O)cc1 |$_R3;;;;;;;;;_R1;;_R2;;;$|",
            "MonomerType": "Backbone",
            "PolymerType": "PEPTIDE",
            "NaturalAnalog": "Y",
            "MonomerName": "Tyrosine",
            "Attachments": [{
                    "AttachmentID": "R1-H",
                    "AttachmentLabel": "R1",
                    "CapGroupName": "H",
                    "CapGroupSmiles": "[*][H] |$_R1;$|"
                },
                {
                    "AttachmentID": "R2-OH",
                    "AttachmentLabel": "R2",
                    "CapGroupName": "OH",
                    "CapGroupSmiles": "O[*] |$;_R2$|"
                },
                {
                    "AttachmentID": "R3-H",
                    "AttachmentLabel": "R3",
                    "CapGroupName": "H",
                    "CapGroupSmiles": "[*][H] |$_R3;$|"
                }
            ]
        }
