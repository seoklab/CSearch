import os, sys
import glob
import random
import numpy as np
from scipy.spatial import distance as D
from rdkit import Chem
from openbabel import pybel
from openbabel.pybel import readfile, readstring

chembl_path = '/home/hakjean/galaxy2/test2/chembldata/chembl_27_sdf/nonhsmiles/'

file_list = glob.glob(os.path.join(chembl_path, '*smi'))
file_list_shuffled = random.Random(31).shuffle(file_list)
first = True
near_molecule = False

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


#print(file_list)
out=[]
with open('initial_bank_0531.smi', 'a') as outfile:
    for fn in file_list:
        with open(fn, 'r') as fil:
            for line in fil:
                if line.find('-') != -1 or line.find('+') != -1:
                    continue
                if first:
                    outfile.write(line)
                    out.append(line)
                    first = False
                else:
                    b = line.split()
                    for i in out:
                        a = i.split()
                        if calc_tanimoto_distance(a[0], b[0]) < 0.3:
                            near_molecule = True
                            break
                    if near_molecule:
                        break
                    outfile.write(line)
                    out.append(line)
                    near_molecule = False
        if len(out) ==48:
            break
