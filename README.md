# RecNet_NeuralCodes
 
[![bioRXiv shield](https://img.shields.io/badge/arXiv-1709.01233-red.svg?style=flat)](https://www.biorxiv.org/content/10.1101/2020.11.18.389197v2)


- [Overview](#overview)
- [License](#License)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)

# Overview
This code simulates a recurrent neural network which uses the structured recurrent connections and internal Poisson spiking variability to implement
Gibbs sampling on the stimulus feature subspace in neural response space to approximate the joint posterior over a stimulus and a context feature. 
We use this network model to show the network has an optimal non-zero recurrent strength to correctly samplg the joint posterior, and the optimal recurrent stregnth is determined by the correlation strength between stimulus and context in the world.

The network has excitatory (E) and inhibitory (I) neurons, with each neuron modeled as a Hawkes process. 
The E neurons have Gaussian tunings over the stimulus feature, while the I neurons don't have selectivity over stimulus and only keep the stability of the network.

Recurrent connections in the network

- Two E neurons are connected according to their tuning similarity.
- All other connections in the network, including E to I, I to E and I to I are unstructured.

The network model have three sources of variability:

- Feedforward Poisson inputs.
- Internal Poisson spiking variability.
- Recurrent Poisson-like variability.

More details of the network model and the results of this code can be found at our bioRxiv paper at:
https://www.biorxiv.org/content/10.1101/2020.11.18.389197v2

# System Requirements
The code was developed on MATLAB R2018b and can be ran on MATLAB with new version.

The `simExpMdls.m` can be ran on a standand computer with enough RAM. The computer should have at least 8GB of RAM.

The `Cluster_code\simNet_ScanPars.m` should be ran on a high performance cluster (HPC). 
The user needs to revise `simNet_ScanPars.m` based on the job scheduler or mangement on the HPC.


# License
This project is covered under the **Apache 2.0 License**.
