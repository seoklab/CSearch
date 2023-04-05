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


_models = {}


def _init_model(pdbid: str):
    a = torch.cuda.current_device()
    device = set_device(use_gpu=True, gpu_idx=a)

    model = MyModel(
        model_type='gcn',
        num_layers=4,
        hidden_dim=128,
        readout='pma',
        dropout_prob=0.2,
        out_dim=2,
    )
    model = model.to(device)

    if pdbid == "4DJQ":
        save_path = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/4djq_gcn_128_pma_m2cdo.pth'
    elif pdbid == "5P9H":
        save_path = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/5p9h_gcn_128_pma.pth'
    elif pdbid == "6M0K":
        save_path = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/free_workplace/opps/save/MCDO_gcn_128_pma_mcdo.pth'
    else:
        raise ValueError('Invalid pdbid: you should choose pdbid in [4DJQ, 5P9H, 6M0K]')

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


def sa_calc(input_mols):
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer

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
    a = torch.cuda.current_device()
    device = set_device(use_gpu=True, gpu_idx=a)

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
                for _ in range(0, 10):
                    tmp_graph = graph.clone()
                    pred, _ = model(tmp_graph)
                    pred = pred.unsqueeze(-1)
                    tmp_list.append(pred)

                tmp_list = torch.cat(tmp_list, dim=-1)
                mean_list = torch.mean(tmp_list, dim=-1)

                pred_list.append(mean_list[:, 0])

        pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
        pred_list = list(np.around(pred_list, 3))
        galigandE = pred_list
        return galigandE

    if input_file == "single":
        # Set random seeds and device
        set_seed(seed=621)
        device = set_device(
            use_gpu=True,
            gpu_idx=1
        )

        # Call smiles from input
        smiles_in = input_mols
        #print(smiles_in)
        if smiles_in.find('~') != -1:

            smiles_in.replace('~','')
        test_ds = MyDataset(smi_list=[smiles_in])
        test_loader = DataLoader(
            dataset=test_ds,
		    batch_size=64,
		    shuffle=False,
		    num_workers=8,
		    collate_fn=my_collate_fn
    	)

        #Construct model and load trained parameters if it is possible
        reo = 'pma'
        if input_pdbid == "5P9H":
            reo = 'mean'

        model = MyModel(
            model_type = 'gcn',
            num_layers=4,
		    hidden_dim=128,
		    readout=reo,
		    dropout_prob=0.2,
		    out_dim=2,
        )
        model = model.to(device)

        if input_pdbid == "4DJQ":
            save_path = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/4djq_gcn_128_pma_m2cdo.pth'
        elif input_pdbid == "5P9H":
            #save_path = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/5p9h_gcn_128_pma.pth'
            save_path = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/5p9h_mean.pth'
        elif input_pdbid == "6M0K":
            save_path = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/free_workplace/opps/save/MCDO_gcn_128_pma_mcdo.pth'
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        #ff = open('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/free_workplace/opps/results/csv_workingplace.csv','w', newline='')

        with torch.no_grad():
            pred_list = []
            smi_list = []
            for i, batch in enumerate(test_loader):
                st = time.time()

                tmp_list = []
                graph_tmp = batch[0]
                smi = batch[1]
                for _ in range(0,10):
                    graph = graph_tmp
                    graph = graph.to(device)
                    pred, alpha = model(graph, training=True)
                    pred = pred.unsqueeze(-1)
                    tmp_list.append(pred)

                tmp_list = torch.cat(tmp_list, dim=-1)
                mean_list = torch.mean(tmp_list, dim=-1)

                pred_list.append(mean_list[:,0])

                smi_list.extend(smi)



        pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()



        pred_list = list(np.around(pred_list, 3))
        galigandE = pred_list[0]
	    #smi_list
	    #pred_list






        return galigandE

    else:
        raise NotImplementedError
    #     smiles_in = input_check(input_files)
    #     calc = Calculator(descriptors, ignore_3D=True)
    #     m = Chem.MolFromSmiles(smiles_in)
    #     descriptor = []
    #     with open('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/data/wor.txt','r') as gi:
    #         for line in gi:
    #             f = line.split(',')
    #     f = np.array(list(map(int,f)))
    #     mordred = np.array(list(calc(m)))
    #     descriptor = mordred[f]
    #     rege_input = np.array(descriptor)
    #     rege_input = rege_input.reshape(1,-1)
    #     galigandE = rege.predict(rege_input)

    #     return galigandE[0]


# def main():
#     print('start',time.ctime())
#     input_files = input_file
#     smiles_in = input_check(input_files)
#     calc = Calculator(descriptors, ignore_3D=True)
#     m = Chem.MolFromSmiles(smiles_in)
#     descriptor = []
#     with open('/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/data/wor.txt','r') as gi:
#         for line in gi:
#             f = line.split(',')
#     f = np.array(list(map(int,f)))
#     mordred = np.array(list(calc(m)))
#     descriptor = mordred[f]
#     rege_input = np.array(descriptor)
#     rege_input = rege_input.reshape(1,-1)
#     galigandE = rege.predict(rege_input)

#     print(galigandE)
#     print('end',time.ctime())
# if __name__ == "__main__":
#     main()


#input file
#smi
