import time
import os,sys
import argparse
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import *
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdMolDescriptors import *
from mordred import Calculator, descriptors
import joblib

rege = joblib.load('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/data/rf_hj_pd_736_1695_07.pkl')
parser = argparse.ArgumentParser()
parser.add_argument('--i', default='smi')
parser.add_argument('--i_file', default='/home/hakjean/test/test2/chembldata/chembl_27_sdf/nonhsmiles/CHEMBL58016.smi')
args = parser.parse_args()
input_type = args.i
input_file = args.i_file

#global input_type
#input_type = 'smi'




def input_check(input_files):
    if input_type == 'smi':
        smi_f = input_file
        with open(smi_f,'r') as g:
            for line in g:
                smile = line.split()
                smiles = smile[0]
    elif input_type == 'sdf':
        os.system('obabel -isdf %s -osmi -O~/galaxy2/developments/MolGen/MolGenCSA/data/%s' % (input_file,'energy'+'.smi'))
        with open('~/galaxy2/developments/MolGen/MolGenCSA/data/energy.smi', 'r') as f:
            for line in f:
                smile = line.split()
                smiles = smile[0]
        os.system('rm ~/galaxy2/developments/MolGen/MolGenCSA/data/energy.smi')

    return smiles


def energy_calc(input_smiles,input_file):
    if input_file == 'csa':
        smiles_in = input_smiles
        calc = Calculator(descriptors, ignore_3D=True)
        m = Chem.MolFromSmiles(smiles_in)
        descriptor = []
        with open('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/data/wor.txt','r') as gi:
            for line in gi:
                f = line.split(',')
        f = np.array(list(map(int,f)))
        mordred = np.array(list(calc(m)))
        descriptor = mordred[f]
        rege_input = np.array(descriptor)
        rege_input = rege_input.reshape(1,-1)
        galigandE = rege.predict(rege_input)
    
        return galigandE[0] 
    else:
        smiles_in = input_check(input_files)
        calc = Calculator(descriptors, ignore_3D=True)
        m = Chem.MolFromSmiles(smiles_in)
        descriptor = []
        with open('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/data/wor.txt','r') as gi:
            for line in gi:
                f = line.split(',')
        f = np.array(list(map(int,f)))
        mordred = np.array(list(calc(m)))
        descriptor = mordred[f]
        rege_input = np.array(descriptor)
        rege_input = rege_input.reshape(1,-1)
        galigandE = rege.predict(rege_input)
        
        return galigandE[0]


def main():
    print('start',time.ctime())
    input_files = input_file
    smiles_in = input_check(input_files)
    calc = Calculator(descriptors, ignore_3D=True)
    m = Chem.MolFromSmiles(smiles_in)
    descriptor = []
    with open('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/data/wor.txt','r') as gi:
        for line in gi:
            f = line.split(',')
    f = np.array(list(map(int,f)))
    mordred = np.array(list(calc(m)))
    descriptor = mordred[f]
    rege_input = np.array(descriptor)
    rege_input = rege_input.reshape(1,-1)
    galigandE = rege.predict(rege_input)

    print(galigandE)
    print('end',time.ctime())
if __name__ == "__main__":
    main()


#input file
#smi



