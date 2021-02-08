The Genetic Algorithm for Research and Development on Nanoclusters (Organisms) Program: A Genetic Algorithm for Nanoclusters
=========================================================================================================================

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Organisms)
![PyPI](https://img.shields.io/pypi/v/Organisms)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/GardenGroupUO/Organisms)
[![Documentation Status](https://readthedocs.org/projects/organisms/badge/?version=latest)](https://organisms.readthedocs.io/en/latest/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GardenGroupUO/Organisms_Live_Examples/main?filepath=Organisms_Live_Example.ipynb)
![GitHub](https://img.shields.io/github/license/GardenGroupUO/Organisms)

The Otago Research Genetic Algorithm for Nanoclusters, Including Structural Methods and Similarity (Organisms) program is designed to perform a genetic algorithm global optimisation for nanoclusters. It has been designed with inspiration from the Birmingham Cluster Genetic Algorithm and the Birmingham Parallel Genetic Algorithm from the Roy Johnston Group (see ``J. B. A. Davis, A. Shayeghi, S. L. Horswell, R. L. Johnston, Nanoscale, 2015,7, 14032`` ([https://doi.org/10.1039/C5NR03774C](https://doi.org/10.1039/C5NR03774C) or [link to pdf here](https://pubs.rsc.org/en/content/articlepdf/2015/nr/c5nr03774c)), ``R. L. Johnston,Dalton Trans., 2003, 4193–4207`` ([https://doi.org/10.1039/B305686D](https://doi.org/10.1039/B305686D) or [link to pdf here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.124.6813&rep=rep1&type=pdf)

This program has been designed to learn about how to improve the efficiency of the genetic algorithm in locating the global minimum. This genetic algorithm implements various predation operators, fitness operators, and epoch methods. A structural comparison method based on the common neighbour analysis (CNA) has been implemented into a SCM-based predation operator and ''structure + energy'' fitness operator. 

The SCM-based predation operator compares the structures of clusters together and excludes clusters from the population that are too similar to each other. This can be tuned to exclude clusters that are structurally very similar to each other, to exclude clusters that are structurally different but of the same motif, or set to a custom structural exclusion setting. 

The ''structure + energy'' fitness operator is designed to include a portion of structural diversity into the fitness value as well as energy. The goal of this fitness operator is to guide the genetic algorithm around to unexplored areas of a cluster's potential energy surface. 

This genetic algorithm has been designed with Atomic Simulation Environment (ASE, [https://wiki.fysik.dtu.dk/ase/](https://wiki.fysik.dtu.dk/ase/)). with the use of ASE, clusters that are generated using the genetic algorithm are placed into databases that you can assess through the terminal or via a website. See more about how to the ASE database works [in the link here](https://wiki.fysik.dtu.dk/ase/ase/db/db.html?highlight=databases#id9).

All the information about this program is found online at [organisms.readthedocs.io/en/latest/](https://organisms.readthedocs.io/en/latest/). It is recommended to read the installation page before using the algorithm ([organisms.readthedocs.io/en/latest/Installation.html](https://organisms.readthedocs.io/en/latest/Installation.html)). Note that you can install Organisms through pip (Organisms). See the instruction about how to do this. 

Have fun!