.. _design_protocol:

Design Protocol
================

In this tutorial, our focus will be the design protocol. In essence, the design protocol provides 
a set of guidelines that govern the optimization process. It delineates the sequence space by 
establishing constraints and implementing filters, thus enabling you to seamlessly integrate 
any pre-existing knowledge about a specific system.

The design protocol is formatted as a YAML document and consists of two primary components: `designs`
and `filters`. The `design` component delineates the sequence space that can be accessed during the 
optimization process. On the other hand, the `filters` component allows you to exclude peptides with 
undesirable properties (such as low solubility or permeability), which will be enforced at the 
end of the optimization process.

Here is an example of a design protocol:

.. code-block:: YAML

    design:
      monomers: 
        default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
        APOLAR: [A, F, G, I, L, P, V, W]
        POLAR: [C, D, E, H, K, N, Q, R, K, S, T, M]
        AROMATIC: [F, H, W, Y]
        POS_CHARGED: [K, R]
        NEG_CHARGED: [D, E]
      polymers:
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

Designs
-------

**Monomers**
The initial section is dedicated to defining unique collections of monomers. In the provided example
above, various collections were established to categorize amino acids with similar characteristics. 
For instance, the APOLAR collection exclusively encompasses hydrophobic amino acids, whereas the 
`POLAR` collection strictly includes polar amino acids.

You have the liberty to define your own custom collections of monomers. For instance, if you wish to 
utilize only your preferred amino acids (since we all have our favorite amino acids!), you can accomplish 
this as demonstrated below:

.. code-block:: YAML

    monomers: 
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
      AA_I_really_love: [E, L, I, N]

In the example above, we defined a collection called `AA_I_really_love` that includes only the amino 
acids E, L, I, N.

When it comes to the default collection, it has a unique role. The default collection is utilized when 
no specific collection or instructions are defined for a certain position in the scaffold (explained 
in more detail below). Nonetheless, the default collection can be adjusted to exclude certain amino acids. 
For instance, the amino acid cysteine, known for forming disulfide bonds, can be excluded as shown in 
the following example:

.. code-block:: YAML

    monomers: 
      default: [A, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]

**Scaffolds**
The second section of the design protocol is dedicated to defining the scaffolds that will be employed 
during the optimization. A scaffold is expressed using the HELM notation (Hierarchical Editing Language 
for Macromolecules). For more information on the HELM notation, please refer to the documentation from 
the `Pistollia Alliance <https://www.pistoiaalliance.org/helm-notation/>`_ to know more about the HELM 
notation. 

For instance, if the objective is to optimize a linear peptide consisting of 9 amino acids, the scaffold can 
be defined in the following way:

.. code-block:: YAML

  design:
    monomers:
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
    polymers:
      - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0

In the provided example, the second position is predetermined to be a methionine (`M`) throughout the 
optimization, whereas positions represented by `X` can be assigned any of the monomers outlined in the 
default collection. Naturally, if you wish to simultaneously optimize peptides with differing scaffolds, 
you can define multiple scaffolds in the following manner:

.. code-block:: YAML

  design:
    monomers:
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
    polymers:
      - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0
      - PEPTIDE1{X.M.X.X.X.X.X.X.X.X}$$$$V2.0

There may be occasions when you wish to optimize a peptide, and you already know that a specific position 
should only contain negatively charged amino acids. In such cases, the scaffolds can be defined as shown 
in the subsequent example:

.. code-block:: YAML

  design:
    monomers:
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
      NEG_CHARGED: [D, E]
    polymers:
      - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0:
        PEPTIDE1:
          4: NEG_CHARGED
      - PEPTIDE1{X.M.X.X.X.X.X.X.X.X}$$$$V2.0:
        PEPTIDE1:
          4: [D, E]
      - PEPTIDE1{X.X.X.X.X.X.X.X.X.X.X}$$$$V2.0:
        PEPTIDE1:
          2: M
          4: [D, E]

In the given example, the fourth position (based on a 1-index system) in both scaffolds is set to be either 
`D` or `E` during the optimization. Notice how in one instance we used the predefined collection (`NEG_CHARGED`), 
and in the other, we directly employed a list of amino acids. In the final scaffold, we also set the second 
position to be methionine (`M`). All of these examples are equivalent and will perform identically 
during the optimization process.

Lastly, it's possible to use multiple collections or specific amino acids for a particular position, as 
demonstrated in the initial example. Here is a corresponding illustration:

.. code-block:: YAML

  design:
    monomers:
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
      NEG_CHARGED: [D, E]
      POS_CHARGED: [K, R]
    polymers:
      - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0:
        PEPTIDE1:
          4: [NEG_CHARGED, POS_CHARGED, H]

Filters
-------

The final section of the design protocol is dedicated to defining filters. Filters are used to exclude
peptides with undesirable properties. For instance, if you wish to exclude peptides with low solubility,
you can employ the `PeptideSolubilityFilter` as shown in the following example:

.. code-block:: YAML

  design:
    monomers:
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
    polymers:
      - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0
  filters:
    - class_path: mobius.PeptideSolubilityFilter
      init_args:
        hydrophobe_ratio: 0.5
        charged_per_amino_acids: 5

In the example above, we defined a filter that excludes peptides with a hydrophobic ratio greater than 0.5
and a charge per amino acid ratio greater than 5. The `init_args` section is used to pass arguments to the
filter. In this case, we passed the `hydrophobe_ratio` and `charged_per_amino_acids` arguments to the
`PeptideSolubilityFilter` filter. For more information on the available filters and their arguments, please
refer to the :ref:`Mobius documentation <mobius>`.

If you want to use multiple filters, you can define them as shown in the following example:

.. code-block:: YAML

  design:
    monomers:
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
    polymers:
      - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0
  filters:
    - class_path: mobius.PeptideSolubilityFilter
      init_args:
        hydrophobe_ratio: 0.5
        charged_per_amino_acids: 5
    - class_path: mobius.PeptideSelfAggregationFilter

In the example above, we defined two filters: `PeptideSolubilityFilter` and `PeptideSelfAggregationFilter`.
The `PeptideSelfAggregationFilter` filter excludes peptides with a propensity to self-aggregate. 
For more information on this particular filter, please refer to the :ref:`Mobius documentation <mobius>`.

In the case you want to implement a custom filter, you can do so by defining a new python class
in a file named for example `myfilter.py`. For instance, we wish to implement a filter that excludes 
peptides that contain more than two consecutives `R`, you can do so as shown in the following example:

.. code-block:: python

  import re
  import numpy as np
  from mobius.utils import parse_helm


  class RemovePeptidesWithRRmotif():

    def __init__(self, **kwargs):
        pass

    def apply(self, polymers):
        p = re.compile('[R]{2,}')
        passed = np.ones(shape=(len(polymers),), dtype=bool)

        for i, complex_polymer in enumerate(polymers):
            simple_polymers, connections, _, _ = parse_helm(complex_polymer)
            
            for _, simple_polymer in simple_polymers.items():        
                if p.search(''.join(simple_polymer)):
                    passed[i] = False
                    break

        return passed

In the example above, we defined a custom filter called `RemovePeptidesWithRRmotif`. The `apply` method
is used to apply the filter to a list of polymers. The `apply` method returns a boolean array, where `True`
indicates that the corresponding polymer passed the filter, and `False` indicates that the corresponding
polymer failed the filter. Now, you can use this filter in your design protocol as shown in the following 
example:

.. code-block:: YAML

  design:
    monomers:
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
    polymers:
      - PEPTIDE1{X.X.X.X.X.X.X.X.X}$$$$V2.0
  filters:
    - class_path: myfiler.RemovePeptidesWithRRmotif

And voilà, you have successfully implemented a custom filter!
