.. _custom_features:

Add custom fingerprints
=======================

Navigating the world of cheminformatics and computational chemistry, you'll often 
bump into the topic of choosing the right molecular representation. It's a 
bit like choosing the right tool for a job - which heavily depends on 
the task you're tackling! One straightforward strategy is to play around with 
different representations, trying them on and see which one fits your task 
the best and yields the most optimal performance.

In the next part of our journey, we're going to roll up our sleeves and dive into 
how you can blend your own features and fingerprints with a Gaussian Process Regression 
model, all with the help of our friendly Python companion, the molfeat package. For 
those craving a deeper understanding of molfeat, feel free to pop over to the 
`molfeat documentation <https://molfeat-docs.datamol.io/stable/>`_.

To install the package, you can use the following command:

.. code-block:: bash

    mamba activate mobius
    mamba install -c conda-forge molfeat


One of the delightful aspects of the `molfeat` package is its extensive wardrobe of 
ready-to-use features and fingerprints. It's like a treasure trove waiting to be 
explored. Check out the list below to see what's in store:

.. code-block:: python

    from molfeat.calc import FP_FUNCS
    print(FP_FUNCS.keys())
    # dict_keys(['maccs', 'avalon', 'ecfp', 'fcfp', 'topological', 'atompair', 
    # 'rdkit', 'pattern', 'layered', 'map4', 'secfp', 'erg', 'estate', 'avalon-count', 
    # 'rdkit-count', 'ecfp-count', 'fcfp-count', 'topological-count', 'atompair-count'])


Leveraging the Map4Fingerprint class as a template (source code available 
`here <https://git.scicore.unibas.ch/schwede/mobius/-/blob/master/mobius/fingerprints.py#L112>`_) 
we can craft our very own fingerprint class. Let's call it `MolFeat`, and create 
it as shown below:

.. code-block:: python

    import numpy as np
    from molfeat.trans.fp import FPVecTransformer
    from rdkit import Chem
    from mobius.utils import MolFromHELM


    class Molfeat:
        def __init__(self, kind='ecfp', input_type='helm_rdkit', dimensions=4096, HELMCoreLibrary_filename=None):
            msg_error = 'Format (%s) not handled. Please use FASTA, HELM_rdkit, HELM or SMILES format.'
            assert input_type.lower() in ['fasta', 'helm_rdkit', 'helm', 'smiles'], msg_error
            # Here we check that the kind of fingerprint requested is available in molfeat
            msg_error = 'This featurizer (%s) in not available. Please use on of these: %s ' % (kind, FP_FUNCS.keys())
            assert kind.split(':')[0] in FP_FUNCS.keys(), msg_error

            # We store the fingerprint kind, the dimensions and the input type
            # Depending on the fingerprint kind, the dimensions might
            # be just ignored, like MACCS.
            self._kind = kind
            self._dimensions = dimensions
            self._input_type = input_type.lower()
            self._HELMCoreLibrary_filename = HELMCoreLibrary_filename

        def transform(self, sequences):
            if not isinstance(sequences, (list, tuple, np.ndarray)):
                sequences = [sequences]

            try:
                if self._input_type == 'fasta':
                    mols = [Chem.rdmolfiles.MolFromFASTA(s) for s in sequences]
                elif self._input_type == 'helm_rdkit':
                    mols = [Chem.rdmolfiles.MolFromHELM(s) for s in sequences]
                elif self._input_type == 'helm':
                    mols = MolFromHELM(sequences, self._HELMCoreLibrary_filename)
                else:
                    mols = [Chem.rdmolfiles.MolFromSmiles(s) for s in sequences]
            except AttributeError:
                print('Error: there are issues with the input molecules')
                print(sequences)

            smiles = [Chem.MolToSmiles(mol) for mol in mols]

            # Now we can use the FPVecTransformer class to transform the 
            # input sequences into fingerprints
            featurizer = FPVecTransformer(self._kind, length=self._dimensions)
            fps = featurizer.transform(smiles)
            fps = np.asarray(fps)

            return fps

And there you have it! We've now created our very own fingerprint class that can 
be used in harmony with the Gaussian Process Regression method. Now, let's take 
it for a spin and see it in action on the MHC class I dataset!

First things first, let's import the necessary packages and functions:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    from mobius import GPModel, TanimotoSimilarityKernel
    from mobius import convert_FASTA_to_HELM, ic50_to_pic50

We then read the MHC class I dataset and split it into training and testing sets:

.. code-block:: python

    # We read first the MHC class I dataset
    # You can find that file at the root of the repository in the data folder
    mhci = pd.read_csv('data/mhc/bdata.20130222.mhci.csv')

    # A lot of peptides were set with those IC50 values. Looks like some default values.
    dirty_values = [1, 2, 3, 5000, 10000, 20000, 43424, 50000, 69444.44444, 78125]

    # Select only 9-mers and removed peptides with these dirty IC50 values.
    mhci = mhci[(mhci['mhc_allele'] == 'HLA-A*02:01') &
                (mhci['length'] == 9) &
                (~mhci['affinity_binding'].isin(dirty_values))].copy()

    # Convert IC50 to pIC50 (a pIC50 of 0 corresponds to an IC50 of 1 nM)
    mhci['pic50'] = ic50_to_pic50(mhci['affinity_binding'])

    # Convert FASTA sequences to HELM format
    mhci['helm'] = convert_FASTA_to_HELM(mhci['sequence'].values)

    # And we split the dataset into training and testing sets
    # We don't use the whole dataset because it takes too long to train the model
    X_train, X_test, y_train, y_test = train_test_split(mhci['helm'][::10].values, 
                                                        mhci['pic50'][::10].values, 
                                                        test_size=0.30, random_state=42)

With everything set up, we're ready to train our Gaussian Process Regression (GPR) 
model using the MolFeat class, employing a variety of fingerprint methods:

.. code-block:: python
    
    fp_methods = ['ecfp', 'avalon', 'maccs', 'fcfp', 'secfp', 'rdkit']

    kernel = TanimotoSimilarityKernel()

    for fp_method in fp_methods:
        mlfp = Molfeat(kind=fp_method, dimensions=4096)
        gpmodel = GPModel(kernel=kernel, transform=mlfp)
        gpmodel.fit(X_train, y_train)
        mu, _ = gpmodel.predict(X_test)
        print(f'{fp_method} -- '
              f'r2: {r2_score(y_test, mu):.3f} - '
              f'RMSD: {np.sqrt(mean_squared_error(y_test, mu)):.3f}')
    
    # ecfp -- r2: 0.384 - RMSD: 1.075
    # avalon -- r2: 0.311 - RMSD: 1.136
    # maccs -- r2: 0.191 - RMSD: 1.231
    # fcfp -- r2: 0.406 - RMSD: 1.055
    # secfp -- r2: 0.374 - RMSD: 1.084
    # rdkit -- r2: 0.312 - RMSD: 1.136

As is evident, the performance varies among different fingerprint methods. The
MACCS fingerprint method turns out to be the underperformer here, scoring an 
R^2 of 0.191 and an RMSD of 1.231. On the other hand, the fcfp method puts 
up a more impressive show with an R^2 of 0.406 and an RMSD of 1.055, however
it's still not that great. Clearly, there is some room for improvement!
