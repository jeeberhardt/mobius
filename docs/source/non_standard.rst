.. _non_standard:

Macrocycles and non-standard monomers
=====================================

Are you weary of those flashy, `Selling Sunset` methods that promise the world 
but fall short, functioning solely on linear peptides with the same old 20 
standard-amino acids? I feel you. It's like being served a bland meal after 
reading a mouth-watering menu. That's exactly why `mobius` was created. Designed 
to bring the zest back into your peptide optimization work, boldly stepping 
outside the box. It natively handles complex peptide scaffolds, embracing 
non-natural amino acids. With `mobius`, the peptide optimization world is 
your oyster, ready to be explored beyond the limitations of standard amino acids.

Ready to dive in? Excellent! Let's roll up our sleeves and get to work. We're 
going to kick things off with a simple (albeit not entirely realistic) example. 
For this exercice, we are going to take a ramdom macrocycle peptides containing 
non-natural amino acids from the `CycPeptMPDB <http://cycpeptmpdb.com/>`_:

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
have to keep in mind that while mobius is a robust tool, it doesn't possess magical 
abilities (at least not yet!). So, we're proceeding with a couple of assumptions:

#. We're already informed about the peptide scaffold (a 10-mer with a link between monomers 4 and 10).
#. We know the non-standard amino acids that need to be integrated into the mix.

To simulate a closed-loop Design-Make-Test optimization cycle, we're going to harness 
the power of the FindMe emulator. This clever tool works by setting up a target 
sequence—in this case, the sequence we defined earlier—and a kernel function, 
specifically the Tanimoto kernel, to evaluate our distance from the target. The end 
goal? To unearth the target sequence.

Since we're employing the Tanimoto metric, scores will range from 0 (indicating a 
completely different sequence) to 1 (indicating an identical sequence).

.. code-block:: python

    # Here we use the (folded) MinHashed Fingerprint method
    mhfp = MHFingerprint(input_type='helm', dimensions=4096, radius=3)
    kernel = TanimotoSimilarityKernel()

    # Here we define the FindMe emulator
    fm = FindMe(target, input_type='helm', kernel=kernel, input_transformer=mhfp)

With the emulator now in place, our next step is to define our lead sequence, which 
only consists of standard amino acids. The Tanimoto score standing between the target 
and the lead sequence is approximately 0.394. Using this as our starting point, 
we'll generate a seed library comprising 96 sequences. These sequences are obtained 
by applying the alanine and homolog scanning methods.

.. code-block:: python

    lead = 'PEPTIDE1{E.P.L.T.A.K.I.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0'

    seed_library = [lead_peptide]

    for seq in alanine_scanning(lead_peptide):
        seed_library.append(seq)

    for seq in homolog_scanning(lead_peptide):
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

    gpmodel = GPModel(kernel=TanimotoSimilarityKernel(), input_transformer=mhfp)
    acq = ExpectedImprovement(gpmodel, maximize=True)
    optimizer = SequenceGA(total_attempts=5, temperature=0.1)

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
      scaffolds:
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

    ps = Planner(acq, optimizer, design_protocol='sampling_macrocycle.yaml')

    peptides = list(seed_library)[:]
    scores = list(scores_seed_library)[:]

    for i in range(5):
        suggested_peptides, _ = ps.recommand(peptides, scores, batch_size=96)

        # Here you can add whatever methods you want to further filter out peptides

        # Virtually test the suggested peptides using the MHC emulator
        # You know the drill now, this is for benchmarking or demonstration
        # purposes only and should be replaced with actual lab experiments.
        scores_suggested_peptides = fm.score(suggested_peptides)

        peptides.extend(list(suggested_peptides))
        scores.extend(list(scores_suggested_peptides))

        best_seq = peptides[np.argmax(scores)]
        best_scores = np.max(scores)
        print('Best peptide found so far: %s / %.3f' % (best_seq, best_scores))
        print('')

Typically, you'd see output similar to the following (excluding all the warnings, of course):

.. code-block:: none

    N 001 (01/05) - Score: 0.003 - PEPTIDE1{V.P.L.T.A.K.F.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 002 (02/05) - Score: 0.003 - PEPTIDE1{V.P.L.T.A.K.F.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 003 (03/05) - Score: 0.003 - PEPTIDE1{V.P.L.T.A.K.F.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 004 (04/05) - Score: 0.003 - PEPTIDE1{V.P.L.T.A.K.F.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 005 (05/05) - Score: 0.003 - PEPTIDE1{V.P.L.T.A.K.F.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Reached maximum number of attempts 5, no improvement observed!
    End SequenceGA - Best score: 0.003 - PEPTIDE1{V.P.L.T.A.K.F.G.L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Best peptide found so far: PEPTIDE1{T.[d1-Nal].F.T.T.[dL].L.[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.601

    N 001 (01/05) - Score: 0.000 - PEPTIDE1{T.[d1-Nal].F.T.T.[dL].L.[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)    
    N 002 (02/05) - Score: 0.000 - PEPTIDE1{T.[d1-Nal].F.T.T.[dL].L.[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 003 (03/05) - Score: 0.000 - PEPTIDE1{T.[d1-Nal].F.T.T.[dL].L.[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 004 (04/05) - Score: 0.000 - PEPTIDE1{T.[d1-Nal].F.T.T.[dL].L.[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 005 (05/05) - Score: 0.000 - PEPTIDE1{T.[d1-Nal].F.T.T.[dL].L.[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Reached maximum number of attempts 5, no improvement observed!
    End SequenceGA - Best score: 0.000 - PEPTIDE1{T.[d1-Nal].F.T.T.[dL].L.[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Best peptide found so far: PEPTIDE1{[ac].T.[Me_dA].T.[Me_dA].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.707

    N 001 (01/05) - Score: 0.002 - PEPTIDE1{[ac].T.[Me_dA].T.[Me_dA].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)    
    N 002 (02/05) - Score: 0.002 - PEPTIDE1{[ac].T.[Me_dA].T.[Me_dA].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 003 (03/05) - Score: 0.002 - PEPTIDE1{[ac].T.[Me_dA].T.[Me_dA].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 004 (04/05) - Score: 0.002 - PEPTIDE1{[ac].T.[Me_dA].T.[Me_dA].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 005 (05/05) - Score: 0.002 - PEPTIDE1{[ac].T.[Me_dA].T.[Me_dA].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Reached maximum number of attempts 5, no improvement observed!
    End SequenceGA - Best score: 0.002 - PEPTIDE1{[ac].T.[Me_dA].T.[Me_dA].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Best peptide found so far: PEPTIDE1{[ac].P.[dL].T.[d1-Nal].[d1-Nal].[Nva].[dL].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.758

    N 001 (01/05) - Score: 0.003 - PEPTIDE1{[ac].P.[dL].T.[Me_dL].[Nva].[d1-Nal].[Nva].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 002 (02/05) - Score: 0.003 - PEPTIDE1{[ac].P.[dL].T.[Me_dL].[Nva].[d1-Nal].[Nva].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 003 (01/05) - Score: 0.006 - PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].L.A.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 004 (02/05) - Score: 0.006 - PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].L.A.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 005 (03/05) - Score: 0.006 - PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].L.A.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 006 (04/05) - Score: 0.006 - PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].L.A.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 007 (05/05) - Score: 0.006 - PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].L.A.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Reached maximum number of attempts 5, no improvement observed!
    End SequenceGA - Best score: 0.006 - PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].L.A.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Best peptide found so far: PEPTIDE1{[ac].P.[Me_dL].T.[d1-Nal].[Nva].[dL].L.A.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.835

    N 001 (01/05) - Score: 0.003 - PEPTIDE1{[ac].[Me_dL].[dL].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 002 (02/05) - Score: 0.003 - PEPTIDE1{[ac].[Me_dL].[dL].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 003 (03/05) - Score: 0.003 - PEPTIDE1{[ac].[Me_dL].[dL].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 004 (01/05) - Score: 0.013 - PEPTIDE1{[ac].P.[Me_dA].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 005 (02/05) - Score: 0.013 - PEPTIDE1{[ac].P.[Me_dA].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 006 (03/05) - Score: 0.013 - PEPTIDE1{[ac].P.[Me_dA].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 007 (04/05) - Score: 0.013 - PEPTIDE1{[ac].P.[Me_dA].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    N 008 (05/05) - Score: 0.013 - PEPTIDE1{[ac].P.[Me_dA].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Reached maximum number of attempts 5, no improvement observed!
    End SequenceGA - Best score: 0.013 - PEPTIDE1{[ac].P.[Me_dA].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 (10)
    Best peptide found so far: PEPTIDE1{[ac].P.[Me_dA].T.[d1-Nal].[Nva].[dL].[Me_dA].L.P}$PEPTIDE1,PEPTIDE1,4:R3-10:R2$$$V2.0 / 0.922


As you can see, while we didn't completely nail it, we got extremely close! The closest 
sequence discovered bears a Tanimoto score of 0.922 when compared with the target sequence.
This result is encouraging as it illustrates that the method is effective. However, 
it also highlights that there's still room for significant improvements. Let's consider 
this a successful starting point and a call to further optimize our approach!

.. warning::

    Before we wrap up, it's important to note that this experiment is a simplified scenario 
    and it might differ from real-world applications. In actual experiments, your results may 
    contain uncertainties, and more challenging still, you might not obtain a clear outcome for 
    every peptide tested. However, don't let this discourage you! These challenges make the 
    field of peptide optimization a dynamic and fascinating area to explore. 
