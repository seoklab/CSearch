import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from rdkit.Chem.Descriptors import *
from rdkit.Chem.rdMolDescriptors import *

from .libs.models import MyModel
from .libs.io_inference import MyDataset, my_collate_fn
from .libs.utils import set_seed, set_device

parser = argparse.ArgumentParser()
parser.add_argument('--i', default='smi')
parser.add_argument('--i_file', default='/home/hakjean/test/test2/chembldata/chembl_27_sdf/nonhsmiles/CHEMBL58016.smi')
args = parser.parse_args()
input_type = args.i
input_file = args.i_file


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


def energy_calc(input_smiles, input_file, model):
    if input_file == 'csa':
        # Set random seeds and device
        set_seed(seed=621)
        device = set_device(
            use_gpu=True,
            gpu_idx=1
        )

        # Call smiles from input
        smiles_in = input_smiles
        #print(smiles_in)
        #for ii in smiles_in:
        #    if ii.find('~') != -1:

        #        ii.replace('~','')
        test_ds = MyDataset(smi_list=smiles_in)
        test_loader = DataLoader(
            dataset=test_ds,
		    batch_size=128,
		    shuffle=False,
		    num_workers=8,
		    collate_fn=my_collate_fn
    	)
        #print("test_loader")
        #print(test_loader)
        #Construct model and load trained parameters if it is possible
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
        galigandE = pred_list
	    #smi_list
	    #pred_list






        return galigandE
    if input_file == "single":
        # Set random seeds and device
        set_seed(seed=621)
        device = set_device(
            use_gpu=False,
            gpu_idx=1
        )

        # Call smiles from input
        smiles_in = input_smiles
        print(smiles_in)
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
       #ff = open('uhome/hakjean/galaxy2/developments/MolGen/MolGenCSA/free_workplace/opps/results/csv_workingplace.csv','w', newline='')

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
