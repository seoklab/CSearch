import argparse
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Recap,BRICS,Descriptors
from rdkit.Chem.Descriptors import NumRadicalElectrons
from openbabel import pybel
from openbabel.pybel import readfile, readstring
#from opps.energy_calculation_tab import energy_calc,qed_calc,sa_calc

fp_compl = []
bank_smiles = []
dist_compl = []
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_fp_array(smi_list, fty):
    for i in smi_list:
        mol = str(i)
        if mol.find("~") != -1:
            mol = mol.replace("~","")
        mol = pybel.readstring("smi",mol)
        fp = mol.calcfp(fptype=fty)
        a = str(fp)
        b = a.split(',')
        for j in range(0,len(b)):
            b[j] = int(b[j])
        arr = np.array(b)
        fp_compl.append(arr)
    return fp_compl

def make_tsne_xy(fp_array):
    X_embedded = TSNE(n_components=2,learning_rate='auto', init='random',random_state=21,perplexity=10).fit_transform(fp_array)
    x = []
    y = []
    n = len(fp_array)
    for u in range(0,n):
        x.append(X_embedded[u][0])
    for jj in range(0,n):
        y.append(X_embedded[jj][1])
    X=np.array(x)
    Y=np.array(y)
    colors = np.random.rand(n)
    plt.scatter(X,Y,s=1,c=colors)
    plt.title("t-sne")
    plt.savefig(f'/home/hakjean/pictures/t-sne_plot_{n}.png', dpi=200)


def main(args):
    #if args.smi == None:
    #    print('Please Write the smi file with its path')
    with open(f'{args.smi}','r') as g:
        for line in g:
            smiles = line.split()
            smiles = smiles[0]
            bank_smiles.append(smiles)
    compl = make_fp_array(bank_smiles)
    n = len(compl)
    compl = np.array(compl)
    if args.label:
        energy_pool = energy_calc(bank_smiles, "csa", args.pdbid) 
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random',random_state=21,perplexity=3).fit_transform(compl)
    x = []
    y = []
    for u in range(0,n):
        x.append(X_embedded[u][0])
    for jj in range(0,n):
        y.append(X_embedded[jj][1])
    X=np.array(x)
    Y=np.array(y)
    colors = np.random.rand(n)
    if args.label:
        fig, ax = plt.subplots()
        ax.scatter(X,Y,c=colors)
        for di, txt in enumerate(energy_pool):
            ax.annotate(txt, (X[di],Y[di]))

        ax.title("t-sne")
        ax.axis("tight")
        ax.show()
    else:
        plt.scatter(X,Y,c=colors)
        plt.title("t-sne")
        plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--smi", type=str, required=True, help="SMILES file for t-SNE plot")
    parser.add_argument("-l", "--label", type=str2bool, default=False, help="Show the energy of each point")
    parser.add_argument("-p", "--pdbid", type=str, default='6M0K', help="PDB id for energy calc")
    args = parser.parse_args()
    print("Arguments")
    for k, v in vars(args).items():
        print (k, ": ", v)
    main(args)
    
