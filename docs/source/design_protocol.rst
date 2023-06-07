.. _design_protocol:

Design Protocol
================

In this tutorial, we'll focus on understanding the structure of the design protocol. The 
design protocol contain the set of rules that will applied during the optimization. It will
help definining precisely the design space, set constraints and filters, while also 
allowing you to incorporate prior knowledge about your particular system.

The design protocol is a YAML formatted document, and it is constituted of two main components: 
: the `design` and the `filters`. The `design` component defines the design space during the 
optimization, while the `filters` component helps you exclude peptides with unwanted properties 
(not soluble, not permeable, etc...) that will be applied at the end of the optimization.

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

# Design

## Monomers
The first section is used to define the collections of monomers defined with a unique name. In the 
initial example above, multiple collections were defined to group amino acids with similar
properties. For example, `APOLAR` collection contains only hydrophobic amino acids, while the
`POLAR` collection contains only polar amino acids.

Of course you can define your own collections of monomers. For example, if you want to use only
amino acids you really like (everyone has a favorite amino acid!), you can do it as follows:

.. code-block:: YAML

    monomers: 
      default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
      AA_I_really_love: [E, L, I, N, E]

Concerning the `default` collection, this is a special one. The `default` collection is used when 
no collection is specified for a particular position in the scaffold (see below). However, the 
`default` can be modified to exclude for example the cysteine amino acid, which is known to 
form disulfide bonds. For example:

.. code-block:: YAML

    monomers: 
      default: [A, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]

## Scaffolds
The second section is used to define the scaffolds. The scaffold is defined using the HELM notation 
(Hierarchical Editing Language for Macromolecules). The scaffold is defined as a string, where
`X` is used to define a position with an unknown monomers. For example, the scaffold 
`PEPTIDE1{X.M.X.X.X.X.X.X.X}` defines a peptide with 9 positions, where the second position is
a methionine.




# Filters
