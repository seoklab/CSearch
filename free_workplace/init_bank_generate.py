import os,sys
import glob
import random
import numpy as np
from scipy.spatial import distance as D
from rdkit import Chem
from openbabel import pybel
from openbabel.pybel import readfile
import pandas as pd
from tqdm import tqdm
near_molecule = False
def calc_tanimoto_distance(mol1, mol2):
    mol1 = str(mol1)
    mol2 = str(mol2)
    if mol1.find("~") != -1:
        mol1 = mol1.replace("~","")
    if mol2.find("~") != -1:
        mol2 = mol2.repalce("~", "")
    mol1 = pybel.readstring("smi", mol1)
    mol2 = pybel.readstring("smi", mol2)
    fp1 = mol1.calcfp(fptype="fp4")
    fp2 = mol2.calcfp(fptype="fp4")
    tani = fp1|fp2
    dist = 1.0 - tani
    return dist
out = []
first = True
df = pd.read_csv('energy_sorted2.csv', index_col = 0)
print(df.columns)
smi_list = list(df['smiles'])
with open('initial_bank_0801.smi', 'w') as outfile:
    for i in tqdm(smi_list):
        if first:
            outfile.write(f'{i}\n')
            out.append(i)
            first = False
        else:
            for o in out:
                if calc_tanimoto_distance(i, o) < 0.2:
                    near_molecule = True
                    break
            if near_molecule:
                near_molecule = False
                continue
            outfile.write(f'{i}\n')
            out.append(i)
