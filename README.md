# CSearch: Chemical Space Search via Virtual Synthesis and Global Optimization


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

#### -p --pdbid 
The target protein to optimize docking energy. "6M0K" for MPro, "5P9H" for BTK, "4MKC" for ALK, and "3TI5" for H1N1 NA

#### -i --initial-bank 
The initial bank to start CSearch. Formatted with "data/Initial_bank_PDBID_drugspace.smi" or "data/Initial_bank_PDBID_drugbank.smi"

#### -z --banksize 
The size of the bank. Default value is 60.

#### -b --building-blocks
The fragment set SMILES file for virtual synthesis. 

#### -s --seed_num 
The number of the seed molecules that used in virtual synthesis at first cycle. Default value is 6.

#### -c --seed_cycle 
The number of the seed cycle. Default value is 2.

#### -f --frtrack 
The option of tracking the origin of generated molecules by fragments during CSearch optimization or not.

#### -t --filter
The option of filtering generated molecule by lipinski rule of 5 or not.

#### -n --num 
The index for parallel implement. Default value is 0.

#### -a --ann 
Number of the step that Rcut approaches to the Rmin. Default value is 20.

#### -m --maxiter 
Number of the max iterations. Default value is 50.

#### -d --dmin 
The ratio of Rmin compare to Rcut_initial. Default value is 5(means 1/5).


### Example command

    python CSearch.py -p 6M0K -i data/Initial_bank_6M0K_drugspace.smi -b data/Enamine_Fragment_Collection_single.smi

The results would be saved in Result/6M0K/DATE/Bank60_seed6_sc2_mx50_nst20_pace_5_0

## 2. Code design
###  CSearch.py: The main code of CSearch.
###  run_exploremol.py: The execute code based on slurm.
###  opps/energy_calculation.py: Docking energy evaluation code based on pre-trained GNN.
###  opps/fragment_merge.py: Molecule generation code based on virtual synthesis with fragments.
###  opps/libfilter.py: Code of filters like lipinski rule of 5.
