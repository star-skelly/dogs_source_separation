# DOGS (DOuble GMM Starlet) source separation technique

In the crowded center of the Milky Way, it can be difficult to pick out specific objects for spectral and energy analysis. Blind source separation (BSS) allows for extracting scientifically meaningful observations about astrophysical objects. Traditional BSS methods, struggle to extract compact and smooth separations in busy areas. In this work, we introduce DOGS (DOuble GMM with Starlets), a flexible BSS framework that exploits GMM probability fitting for multiple levels of structure from starlets. The framework first splits background from compact objects, then splits foreground objects into independent components. We use this separation framework to evaluate the spectral properties of distinct physical objects: Sgr A*, a pulsar wind nebula, star cluster, and the plasma environment in which they are embedded. It is generalizable, able to split the pulsar wind nebula from its tail. Our results show the potential of this method for astronomical observations with any energy distribution.

# Preliminary Results

Below is the successful separation of the galactic core using DOGS.

### Input View of the Galactic Core

<img height="200" alt="zoomed in view of the galactic core" src="https://github.com/user-attachments/assets/68f2b537-73f7-4e79-ad7b-2480b1925665" />


### Stage 1 of separation (background split from objects)

<img src="1_bg_sources_split.png" alt="stage 1 separation" height="200"/>


### Stage 2 of separation (objects split from each other)

<img src="2_split_4sources.png" alt="stage 2 separation" height="200"/>


__Source 0:__ tail of the pulsar wind nebula

__Source 1:__ black hole, Sagittarius A*

__Source 2:__ core of the pulsar wind nebula

__Source 3:__ star cluster


## Visible Improvements in Radial Energy Profiles

<img width="1189" height="396" alt="3b772883-0c1e-4023-8883-d18ea62a748d" src="https://github.com/user-attachments/assets/edcfb067-a5a8-400b-9c1f-f2ce16274f98" />

# Authors
Salem Loucks, Lia Corrales, Jamila Taaki, *University of Michigan*, Mayura Balakrishnan, *McGill University*
