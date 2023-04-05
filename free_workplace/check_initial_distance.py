import os,sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Recap,BRICS,Descriptors
from rdkit.Chem.Descriptors import NumRadicalElectrons
from openbabel import pybel
from openbabel.pybel import readfile, readstring


bank_smiles = []
dist_compl = []
def calc_tanimoto_distance(mol1, mol2):
    mol1 = str(mol1)
    mol2 = str(mol2)
    if mol1.find("~") != -1:
        mol1 = mol1.replace("~","")
    if mol2.find("~") != -1:
        mol2 = mol2.replace("~","")
    mol1 = pybel.readstring("smi",mol1)
    mol2 = pybel.readstring("smi",mol2)
    fp1 = mol1.calcfp(fptype="fp4")
    fp2 = mol2.calcfp(fptype="fp4")
    #mol1 = Chem.rdmolfiles.MolFromSmiles(str(mol1))
    #mol2 = Chem.rdmolfiles.MolFromSmiles(str(mol2))
    #fp1 = FingerprintMols.FingerprintMol(mol1)
    #fp2 = FingerprintMols.FingerprintMol(mol2)
    tani = fp1|fp2
    #tani = DataStructs.FingerprintSimilarity(fp1,fp2)
    dist = 1.0-tani
    return dist


def main():
    with open('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/initial_bank96_1130.smi','r') as g:
        for line in g:
            smiles = line
            bank_smiles.append(smiles)
    for i_mol, smi in enumerate(bank_smiles):
        base_smi = bank_smiles[i_mol]
        for i in range(0,96):
            dist = calc_tanimoto_distance(base_smi,bank_smiles[i])
            dist_compl.append(dist)
    dist_np = np.array(dist_compl)
    dist_2d = np.reshape(dist_np, (-1,96))
    print(dist_2d)

if __name__=="__main__":
    main()
