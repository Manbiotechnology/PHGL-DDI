# PHGL-DDI
## Overview
This repository is the source code of our paper "PHGL-DDI: A pre-training based hierarchical graph learning framework for drug-drug interaction prediction".

## Environment Setting
python=3.7.12
<br> pytorch-1.11.0
<br> cuda-11.3
<br> torch_cluster-1.6.0
<br> torch_scatter-2.0.9
<br> torch_sparse-0.6.13
<br> torch_spline_conv-1.2.1
<br> torch-geometric= 2.0.4

## Service
GeForce RTX 4090 GPU

## Dataset Preparation
PubChem dataset ：contains15.56 million unlabeled molecules，each molecule is represented by SMILES. 

DrugBank dataset : contains 191,808 DDI tuples with 1706 drugs, each drug is represented in SMILES.

ChCh-Miner dataset : contains 959 drugs and 33,669 DDIs, with each row of data containing the IDs of the two drugs and their corresponding SMILES.

### If you want to know more about our work, you can refer to the following documents:

[1] Gao Z, Jiang C, Zhang J, et al. Hierarchical graph learning for protein-protein interaction. Nat Commun. 2023;14(1):1093. Published 2023 Feb 25. doi:10.1038/s41467-023-36736-1.

[2] Wang, Y., Wang, J., Cao, Z. et al. Molecular contrastive learning of representations via graph neural networks. Nat Mach Intell4, 279–287 (2022).

