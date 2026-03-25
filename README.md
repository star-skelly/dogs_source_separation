# DOGS (DOuble GMM Starlet) source separation technique

In the crowded center of the Milky Way, it can be difficult to pick out specific objects for spectral and energy analysis. Blind source separation (BSS) allows for extracting scientifically meaningful observations about astrophysical objects. Traditional BSS methods, and more recent methods such as Generalized Morphological Component Analysis, struggle to extract compact and smooth separations in busy areas. In this work, we introduce a sequence of starlet-fitted Gaussian Mixture Models (GMMs), a flexible BSS framework that exploits the adaptable GMM probability fitting for multiple levels of structure from starlets. The framework first splits background from more compact objects and then splits the foreground objects into independent components. We use this separation framework to evaluate the spectral properties of distinct physical objects: Sgr A*, a pulsar wind nebula, star cluster, and the plasma environment in which they are embedded. It is generalizable, even allowing for splitting a pulsar wind nebula from its tail. Our results show the potential of this method for any astronomical observations and allows for extremely flexible fitting to new energy distributions.

# Preliminary Results
Stage 1 of separation (background split from objects)
![alt text](1_bg_sources_split.png)

Stage 2 of separation (objects split from each other)
![alt text](2_split_4sources.png)
Successful source separation of galactic core: 

Source 0: tail of the pulsar wind nebula 
Source 1: black hole, Sagittarius A*
Source 2: core of the pulsar wind nebula
Source 3: star cluster

# Authors
Salem Loucks, Lia Corrales, Jamila Taaki, Mayura Balakrishnan, *University of Michigan*
