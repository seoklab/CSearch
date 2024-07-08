#### CSearch: Chemical Space Search via Virtual Synthesis and Global Optimization


The two key components of computational molecular design are virtually generating molecules and predicting the properties of these generated molecules. This study focuses on an effective method for virtual molecular generation through virtual synthesis and global optimization of a given objective function. By using a pre-trained graph neural network (GNN) objective function to approximate the docking energies of compounds for four target receptors, we were able to generate highly optimized compounds with 300-400 times less computational effort compared to virtual compound library screening. These optimized compounds have similar synthesizability to drug-like molecules in commercial libraries, and their diversity and novelty are comparable to known binders with high potency. This method, called CSearch, can be effectively utilized to generate chemicals optimized for an appropriate objective function. Even with the GNN function approximating docking energies, CSearch could generate molecules with predicted binding poses to the target receptors similar to known inhibitors, demonstrating its effectiveness in generating drug-like binders.


# CSearch
drug-like molecular generation via pre-trained docking score optimization

## 0. Installation
Please clone the CSearch git repository by

    git clone https://github.com/seoklab/CSearch.git

After cloning the repository of CSearch, install conda environment by

    conda env create -f environment.yml



## 1. How to implement CSearch

    python CSearch.py -p PDBID -i INITIAL_BANK_SMILES -z BANK_SIZE -b BUILDING_BLOCKS -s SEED_NUM -c SEED_CYCLE -f FRGMNT_TRCK -t FILTER -a ANNEAL_SCHEDULE -m MAX_ITER -d --DMIN

-p --pdbid 
-i --initial-bank
-z --banksize
-b --building-blocks
-s --seed_num
-c --seed_cycle
-f --frtrack
-t --filter
-a --ann
-m --maxiter
-d --dmin

The results are saved in Result/PDBID/DATE/BankN_seedN_scN_mxN_nstN_pace_DMIN_N

## 2. Code design
###  CSearch.py: .
###  run_exploremol.py: .
###  opps/.py: .
###  opps/.py: .
###  opps/.py: .
###  opps/.py: .
