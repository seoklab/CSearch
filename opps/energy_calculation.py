import os, sys
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
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


def energy_calc(input_mols, input_file, input_pdbid):
    # Set random seeds and device
    set_seed(seed=621)
    device = torch.device('cpu')
    if input_file == 'csa':
        test_ds = MyDataset(input_mols)
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=128,
            shuffle=False,
            collate_fn=my_collate_fn
        )

        model = get_model(input_pdbid)

        with torch.no_grad():
            pred_list = []
            for graph in test_loader:
                tmp_list = []
                graph = graph.to(device)
                for _ in range(0, 3):
                    tmp_graph = graph.clone()
                    pred, _ = model(tmp_graph)
                    pred = pred.unsqueeze(-1)
                    tmp_list.append(pred)

                tmp_list = torch.cat(tmp_list, dim=-1)
                mean_list = torch.mean(tmp_list, dim=-1)
                pred_list.append(mean_list[:, 0])
        pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
        if input_pdbid == '4MKC' or '3TI5' or '5P9H':
            pred_list = list(np.around(pred_list*10,3))
        else:
            pred_list = list(np.around(pred_list, 3))
        galigandE = pred_list
        return galigandE

    else:
        raise NotImplementedError
   