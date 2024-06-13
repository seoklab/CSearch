from collections import defaultdict

import pandas as pd

import torch
import dgl
from torch.nn import functional as F

from rdkit import Chem

from tdc.single_pred import ADME
from tdc.single_pred import HTS
from tdc.single_pred import Tox


ATOM_VOCAB = [
	'C', 'N', 'O', 'S', 'F',
	'H', 'Si', 'P', 'Cl', 'Br',
	'Li', 'Na', 'K', 'Mg', 'Ca',
	'Fe', 'As', 'Al', 'I', 'B',
	'V', 'Tl', 'Sb', 'Sn', 'Ag',
	'Pd', 'Co', 'Se', 'Ti', 'Zn',
	'Ge', 'Cu', 'Au', 'Ni', 'Cd',
	'Mn', 'Cr', 'Pt', 'Hg', 'Pb'
]

_atom_vocab_to_idx = defaultdict(
    lambda: len(_atom_vocab_to_idx) - 1,
    {x: i for i, x in enumerate(ATOM_VOCAB)})

_bond_type_to_idx = {
    k: i for i, k in enumerate([
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ])}


def one_of_k_encoding(x, vocab):
	if x not in vocab:
		x = vocab[-1]
	return list(map(lambda s: float(x==s), vocab))


def get_atom_feature(atom):
    degree = min(atom.GetDegree(), 5)
    atom_idx = min(_atom_vocab_to_idx[atom.GetSymbol()],39)
    features = [
        F.one_hot(torch.tensor(atom_idx),num_classes=len(ATOM_VOCAB)),
        F.one_hot(torch.tensor(degree), num_classes=6),
        F.one_hot(torch.tensor(atom.GetTotalNumHs()), num_classes=5),
        F.one_hot(torch.tensor(atom.GetImplicitValence()), num_classes=6),
        torch.tensor([atom.GetIsAromatic()], dtype=torch.float32)
    ]
    return torch.cat(features)


def get_bond_feature(bond):
    bt = bond.GetBondType()
    features = [
        F.one_hot(torch.tensor(_bond_type_to_idx[bt]),
                  num_classes=len(_bond_type_to_idx)),
        torch.tensor(
            [bond.GetIsConjugated(), bond.IsInRing()], dtype=torch.float32),
    ]
    bond_feature = torch.cat(features)
    return bond_feature


def get_molecular_graph(mol):
    atom_list = mol.GetAtoms()
    bond_list = mol.GetBonds()

    bond_idxs = torch.tensor(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in bond_list])
    bond_idxs = torch.cat([bond_idxs, bond_idxs.flip(1)])
    graph = dgl.graph(tuple(bond_idxs.T), num_nodes=len(atom_list))

    atom_features = torch.stack([get_atom_feature(atom) for atom in atom_list])
    graph.ndata['h'] = atom_features

    bond_features = torch.stack([get_bond_feature(bond) for bond in bond_list])
    bond_features = bond_features.repeat(2, 1)
    graph.edata["e_ij"] = bond_features
    return graph


def get_smi_and_label(dataset):
	smi_list = list(dataset['Drug'])
	label_list = list(dataset['Y'])
	return smi_list, label_list


def my_collate_fn(batch):
	graph_list = [get_molecular_graph(mol) for mol in batch]
	batch = dgl.batch(graph_list)
	return batch


def get_dataset(
		path,
		smi_column='SMILES',
		dropna=False,
	):
	df = pd.read_csv(path)
	if dropna:
		df = df.dropna()
	smi_list = list(df[smi_column])
	return smi_list


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, mols):
		self.mols = mols

	def __len__(self):
		return len(self.mols)

	def __getitem__(self, idx):
		return self.mols[idx]


def debugging():
	data = ADME(
		name='BBB_Martins'
	)
	split = data.get_split(
		method='random',
		seed=999,
		frac=[0.7, 0.1, 0.2],
	)
	train_set = split['train']
	valid_set = split['valid']
	test_set = split['test']

	smi_train, label_train = get_smi_and_label(train_set)
	graph = get_molecular_graph(smi_train[0])


if __name__ == '__main__':
	debugging()
