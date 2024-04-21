.. _about:

About Mobius
============

`Mobius` provides a robust and versatile Bayesian optimization (BO) approach, 
designed to aid in the design and optimization of peptides within a fully 
automated, closed-loop Design-Make-Test (DMT) pipeline context.

Equipped to manage an array of peptide scaffolds, including complex structures 
such as macrocycles, as well as non-natural amino acids using the 
HELM notation, Mobius employs a variety of combinatorial strategies. These 
strategies, which include alanine, random, homolog, and property-based scanning, 
facilitate the generation of an initial batch of peptides, enabling you to 
jumpstart your optimization project. Moreover, Mobius features an array of 
fingerprint methods that are adaptable to a wide spectrum of peptide types.

`Mobius` shines in its flexibility and modularity, allowing for seamless integration 
with custom surrogate models, fingerprint methods, and filters. This ensures 
that `Mobius` can be tailored to meet the specific demands and requirements of 
your peptide optimization projects.

The source code is available under the Apache license at `https://git.scicore.unibas.ch/schwede/mobius <https://git.scicore.unibas.ch/schwede/mobius>`_.

The method is described in the following paper: `J. Eberhardt, A. Lees, M. Lill, T. Schwede. (2023). Combining Bayesian optimization with sequence- or structure-based strategies for optimization of peptide-binding protein. <https://doi.org/10.26434/chemrxiv-2023-b7l81-v2>`_
