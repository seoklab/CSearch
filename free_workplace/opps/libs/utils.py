import argparse
import random

import math
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from rdkit import Chem
from rdkit.Chem import Draw


def str2bool(v):
	if v.lower() in ['yes', 'true', 't', 'y', '1']:
		return True
	elif v.lower() in ['no', 'false', 'f', 'n', '0']:
		return False
	else:
		raise arparse.ArgumentTypeError('Boolean value expected')


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.random.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def set_device(
		use_gpu,
		gpu_idx
	):
	if use_gpu:
		device = torch.device('cuda:'+str(gpu_idx))
		print ("PyTorch version:", torch.__version__)
		print ("PyTorch GPU count:", torch.cuda.device_count())
		print ("PyTorch Current GPU:", device)
		print ("PyTorch GPU name:", torch.cuda.get_device_name(device))
		return device
	else:
		device = torch.device('cpu')
		return device


def sigmoid(x):
	return 1./1.+np.exp(-x)


def calibration(
		label, 
		pred, 
		bins=10
	):

	width = 1.0 / bins
	bin_centers = np.linspace(0, 1.0-width, bins) + width/2

	conf_bin = []
	acc_bin = []
	counts = []
	for	i, threshold in enumerate(bin_centers):
		bin_idx = np.logical_and(
			threshold - width/2 < pred, 
			pred <= threshold + width
		)
		conf_mean = pred[bin_idx].mean()
		conf_sum = pred[bin_idx].sum()
		if (conf_mean != conf_mean) == False:
			conf_bin.append(conf_mean)
			counts.append(pred[bin_idx].shape[0])

		acc_mean = label[bin_idx].mean()
		acc_sum = label[bin_idx].sum()
		if (acc_mean != acc_mean) == False:
			acc_bin.append(acc_mean)

	conf_bin = np.asarray(conf_bin)
	acc_bin = np.asarray(acc_bin)
	counts = np.asarray(counts)

	ece = np.abs(conf_bin - acc_bin)
	ece = np.multiply(ece, counts)
	ece = ece.sum()
	ece /= np.sum(counts)
	return conf_bin, acc_bin, ece


def evaluate_classification(
		y_list,
		pred_list,
	):
	y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
	pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()

	auroc = roc_auc_score(y_list, pred_list)
	_, _, ece = calibration(y_list, pred_list)

	'''
	To calculate metric in the below,
	scores should be presented in integer type
	'''
	y_list = y_list.astype(int)
	pred_list = np.around(pred_list).astype(int)

	accuracy = accuracy_score(y_list, pred_list)
	precision = precision_score(y_list, pred_list)
	recall = recall_score(y_list, pred_list)
	f1 = 2.0 * precision * recall / (precision + recall)
	return accuracy, auroc, precision, recall, f1, ece


def evaluate_regression(
		y_list,
		pred_list,
	):
	y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
	pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()

	mse = mean_squared_error(y_list, pred_list)
	rmse = math.sqrt(mse)
	r2 = r2_score(y_list, pred_list)
	return mse, rmse, r2


def heteroscedastic_loss(
		pred,
		y,
	):
	mean = pred[:,0]
	logvar = pred[:,1]
	
	loss_val = torch.exp(-logvar) * (y - mean)**2 + logvar
	loss_val *= 0.5
	loss_val = torch.mean(loss_val, dim=0)
	return loss_val


def scatter_plot_regression(
		y_list,
		pred_list,
		ale_unc_list,
		epi_unc_list,
		figure_title,
		tot_unc_threshold=3.0,
	):
	y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
	pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
	ale_unc_list = torch.cat(ale_unc_list, dim=0).detach().cpu().numpy()
	epi_unc_list = torch.cat(epi_unc_list, dim=0).detach().cpu().numpy()
	tot_unc_list = epi_unc_list + ale_unc_list

	color_list = []
	num_samples = ale_unc_list.shape[0]

	unc_high_group = []
	unc_low_group = []
	for i in range(num_samples):
		tot_unc = tot_unc_list[i]
		if tot_unc <= tot_unc_threshold:
			unc_low_group.append(i)
		else:
			unc_high_group.append(i)

	fig = plt.figure()
	x_ = [-12.0, 2.0]
	plt.plot(x_, x_, c='k')
	plt.scatter(y_list[unc_low_group], pred_list[unc_low_group], c='b', s=4, alpha=0.5)
	plt.scatter(y_list[unc_high_group], pred_list[unc_high_group], c='r', s=4, alpha=0.5)
	plt.xlabel('Solubility (Dataset)', fontsize=15)
	plt.ylabel('Solubility (Prediction)', fontsize=15)
	plt.tight_layout()
	plt.savefig('./figures/'+figure_title+'_'+str(tot_unc_threshold)+'.png')

	mse_high = mean_squared_error(y_list[unc_high_group], pred_list[unc_high_group])
	rmse_high = math.sqrt(mse_high)
	r2_high = r2_score(y_list[unc_high_group], pred_list[unc_high_group])
	accuracy_high = evaluate_classification_multi2(
		y_list[unc_high_group],
		pred_list[unc_high_group],
	)

	mse_low = mean_squared_error(y_list[unc_low_group], pred_list[unc_low_group])
	rmse_low = math.sqrt(mse_low)
	r2_low = r2_score(y_list[unc_low_group], pred_list[unc_low_group])
	accuracy_low = evaluate_classification_multi2(
		y_list[unc_low_group],
		pred_list[unc_low_group],
	)

	print ("Low uncertainty group, RMSE:", rmse_low, "\t R2:", r2_low, "\t Accuracy:", round(accuracy_low, 3), "# of samples:", len(unc_low_group))
	print ("High uncertainty group, RMSE:", rmse_high, "\t R2:", r2_high, "\t Accuracy:", round(accuracy_high, 3), "# of samples:", len(unc_high_group))
	return


def ece_multi_classification(y_list, pred_list, bins=10):
	bin_bounderies = torch.linspace(0, 1, bins+1)
	bin_lowers = bin_bounderies[:-1]
	bin_uppers = bin_bounderies[1:]

	softmaxes = F.softmax(pred_list, dim=-1)
	confidences, predictions = torch.max(softmaxes, 1)
	accuracies = predictions.eq(y_list)

	ece = torch.zeros(1)
	for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
		in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
		prop_in_bin = in_bin.float().mean()
		if prop_in_bin.item() > 0:
			accuracy_in_bin = accuracies[in_bin].float().mean()
			avg_confidence_in_bin = confidences[in_bin].mean()
			ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
	return ece.numpy()[0]


def evaluate_classification_multi(
		y_list,
		pred_list,
	):
	y_list = torch.cat(y_list, dim=0).detach().cpu()
	pred_list = torch.cat(pred_list, dim=0).detach().cpu()

	ece = ece_multi_classification(
		y_list=y_list,
		pred_list=pred_list,
	)

	y_list = y_list.numpy()
	pred_list = pred_list.numpy()

	pred_list = np.argmax(pred_list, axis=1)
	accuracy = np.equal(y_list, pred_list)
	accuracy = np.mean(accuracy)
	return accuracy, ece


def evaluate_classification_multi2(
		y_list,
		pred_list,
	):
	if type(y_list) != np.ndarray:
		y_list = torch.cat(y_list, dim=0).detach().cpu()
		pred_list = torch.cat(pred_list, dim=0).detach().cpu()

	correct = 0
	for i in range(y_list.shape[0]):
		true = y_list[i]
		pred = pred_list[i]
		if pred > 0.0:
			pred = 0
		elif pred <= 0.0 and pred > -2.0:
			pred = 1
		elif pred <= -2.0 and pred > -4.0:
			pred = 2
		elif pred <= -4.0:
			pred = 3

		if true > 0.0:
			true = 0
		elif true <= 0.0 and true > -2.0:
			true = 1
		elif true <= -2.0 and true > -4.0:
			true = 2
		elif true <= -4.0:
			true = 3

		if true == pred:
			correct += 1
		
	accuracy = correct / y_list.shape[0]
	return accuracy


def plot_attention(
		smi_list,
		attention_list,
		pred_list,
		unc_list,
		prefix,
	):
	pred_list = pred_list.detach().cpu().numpy()
	unc_list = unc_list.detach().cpu().numpy()

	attention_list = attention_list.squeeze()
	attention_list = attention_list.detach().cpu().numpy()
	attention_list = np.mean(attention_list, axis=1)

	mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
	legends = []
	for i, mol in enumerate(mol_list):
		num_atoms = mol.GetNumAtoms()
		atom_list = mol.GetAtoms()
		highlight_atoms = []
		highlight_colors = []
		for j, atom in enumerate(atom_list):
			attn_val = round(attention_list[i, j] * num_atoms, 2) 
			atom.SetProp('atomLabel', str(attn_val))
		legend = 'Pred: ' + str(round(pred_list[i], 2)) + ' '
		legend += 'Unc: ' + str(round(unc_list[i], 2))
		legends.append(legend)

	img = Draw.MolsToGridImage(
		mol_list,
		molsPerRow=5,
		legends=legends,
		subImgSize=(600,600),
	)
	img.save('./figures/' + prefix + '_attn.png')
