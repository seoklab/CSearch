import os, sys
import time
import numpy as np

import torch
import subprocess
import nuri
from molscore import MolScore, MockGenerator
from moleval.metrics.metrics import GetMetrics
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import *
from rdkit.Chem.rdMolDescriptors import *
from .libs.models import MyModel
from .libs.io_inference import MyDataset, my_collate_fn
from .libs.utils import set_seed, set_device

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

_models = {}


def _init_model(pdbid: str):
    device = torch.device('cpu')
    model = MyModel(
        model_type='gcn',
        num_layers=4,
        hidden_dim=128,
        readout='pma',
        dropout_prob=0.2,
        out_dim=2,
    )
    model = model.to(device)
    if pdbid == "3TI5":
        save_path = 'opps/save/3TI5_gcn_128_pma_m2cdo.pth' 
    elif pdbid == "4MKC":
        save_path = 'opps/save/4MKC_gcn_128_pma_m2cdo.pth'
    elif pdbid == "5P9H":
        save_path = 'opps/save/5P9H_gcn_128_pma_m2cdo.pth'
    elif pdbid == "6M0K":
        save_path = 'opps/save/6M0K_gcn_128_pma_m2cdo.pth'
    else:
        raise ValueError('Invalid pdbid: you should choose pdbid in [3TI5, 4MKC, 5P9H, 6M0K]')

    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _models[pdbid] = model


def get_model(pdbid: str):
    try:
        return _models[pdbid]
    except KeyError:
        _init_model(pdbid)
        return _models[pdbid]

def sa_calc(input_mols):
    output_sa = [sascorer.calculateScore(mol) for mol in input_mols]
    output_sa = list(np.around(output_sa, 3))
    return output_sa


def qed_calc(input_mols):
    output_qed = [Chem.QED.qed(mol) for mol in input_mols]
    output_qed = list(np.around(output_qed, 3))
    return output_qed


def energy_calc(input_mols, input_file_path):
    # Set random seeds and device
    scores = []
    input_mols = [Chem.MolToSmiles(i) for i in input_mols]
    for ii, i in enumerate(input_mols):
        with open(f'smi/{ii}_f3.smi','w') as f:
            f.write(i)
        os.chdir('/home/hakjean/galaxy2/developments/MolGen/CSearch_revised/smi/')
        os.system(f"obabel -ismi {ii}_f3.smi -omol2 -O {ii}_f3.mol2 --gen3D")
        if not os.path.exists(f"{ii}_f3.mol2"):
            smi = next(nuri.readstring("smi",i))
            with open(f"{ii}_f3.mol2", 'w') as gt:
                gt.write(nuri.to_mol2(smi))
        
        #os.system(f"/applic/corina/corina -i t=sdf {i}_f.sdf -o t=mol2 {i}_f.mol2")
        ff = str(subprocess.run(["galign",f"{ii}_f3.mol2","/home/hakjean/galaxy2/developments/MolGen/CSearch_revised/data/3TI5_ori_1.mol2"],capture_output=True)).split()
        print(ff)
        energy = float(ff[11][0:6])
            #os.system(f"galign {i}_f.mol2 {input_file_path} -s >> {i}_f.log")
            #with open(f"{i}_f.log", 'r') as g:
            #    for line in g:
            #        f = line.split()
            #        if line[2] == 'Score':
            #            score = line[3]
            #            scores.append(float(score)*(-100))
            
        os.system(f"rm {ii}_f3.smi")
        #os.sytem(f"rm *.mol2")
        scores.append(energy*-100)
            #os.system(f"rm *.smi*")
        #except:
        #    scores.append(0)
        #    continue
        os.system(f'rm {ii}_f3.mol2')
        os.chdir('/home/hakjean/galaxy2/developments/MolGen/CSearch_revised/')
    return scores

        

