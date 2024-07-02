import random
import os, sys
import json
import tempfile
import subprocess as sp
import numpy as np
import math
import gzip
import pickle
import re
from typing import List, Set
from rdkit import Chem
from rdkit.Chem import Recap, BRICS, AllChem, DataStructs, RDConfig, rdMolDescriptors


from .libfilter import check_catalog_filters, check_lipinski_filter

__all__ = ["Molecule",
           "get_dict_from_json_file", "gen_mashup",
           "calc_tanimoto_distance", "fix_smiles"]



sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))


def calcfrgscore(mol):
    data = pickle.load(gzip.open('opps/save/fpscores.pkl.gz'))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    global fscores
    fscores = outDict

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(mol,2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += fscores.get(sfp, -4) * v
    score1 /= nf
    return score1





def calcfrscore(input_mols):
    output_fr = [calcfrgscore(mol) for mol in input_mols]
    output_fr = list(np.around(output_fr, 5))
    return output_fr

def get_dict_from_json_file(fn_json):
    with open(fn_json, 'r') as fp:
        json_dict = json.loads(fp.read())
    return json_dict

def returnwmolist(weight, molist):
    if len(molist) < 2:
       return molist
    if type(weight) == type(None):
        return molist
    w = []
    for i in list(weight):
        w.append(i/weight.sum())
    molarray = np.random.choice(np.array(molist), len(molist), p=w)
    return molarray.tolist()


class Molecule(object):
    fn_func_json = 'opps/save/All_Rxns_functional_groups.json'
    functional_group_dict: dict = {
        fgrp: Chem.MolFromSmarts(smarts)
        for fgrp, smarts in get_dict_from_json_file(fn_func_json).items()}

    ##MOVE MOLECULE TO CORE
    # read molecule information
    def __init__(self, smiles, RDKmol,
                 source=None, build_3d=False, decompose_method='BRICS'):
        self.smiles = smiles
        self.RDKmol = RDKmol
        self.source = source

        self.HasFunctionalGroup = {}
        self.determine_functional_groups()

        self.is_3d: bool = build_3d
        self.mol2_block: List[str] = []
        if build_3d:
            self._build_mol2_3D()

        self.decompose_method = decompose_method
        self._pieces: Set[str] = set()

    @property
    def pieces(self):
        if not self._pieces:
            self._decompose()
        return self._pieces

    @classmethod
    def from_smiles(cls, smiles: str, source=None, build_3d=False):
        RDKmol = Chem.MolFromSmiles(smiles)
        return cls(smiles, RDKmol, source=source, build_3d=build_3d)

    @classmethod
    def from_rdkit(cls, RDKmol, source=None, build_3d=False):
        smiles = Chem.MolToSmiles(RDKmol)
        return cls(smiles, RDKmol, source=source, build_3d=build_3d)

    def update(self, rdkmol):
        self.RDKmol = rdkmol
        self.smiles = Chem.MolToSmiles(self.RDKmol)
        if self.is_3d:
            self._build_mol2_3D()

        self.determine_functional_groups()
        self._decompose()

    def __repr__(self):
        return self.smiles

    def _build_mol2_3D(self):
        ret = sp.run(["corina", '-i', "t=smiles", "-o", "t=mol2", "-t", "n"],
                     input=self.smiles, stdout=sp.PIPE, check=True, text=True)
        self.mol2_block = [line for line in ret.stdout.splitlines()
                           if not line.startswith('#')]
        if not self.mol2_block:
            raise ValueError("cannot generate 3d structure")

    def build_3D(self, method='corina', rebuild=False):
        if method != 'corina':
            raise NotImplementedError

        if not rebuild and self.is_3d:
            return

        self.is_3d = True
        self._build_mol2_3D()

    def _decompose(self):
        # for BRICS retrosynthesis
        if self.decompose_method == 'BRICS':
            pieces = BRICS.BRICSDecompose(
                self.RDKmol, minFragmentSize=3,
                singlePass=True, keepNonLeafNodes=True)
        elif self.decompose_method == 'RECAP':
            pieces = Recap.RecapDecompose(self.RDKmol)
        else:
            raise NotImplementedError

        self._pieces = set(pieces)

    def determine_functional_groups(self):
        # for in-silico reaction
        self.HasFunctionalGroup = {
            fgrp: self.RDKmol.HasSubstructMatch(substruct)
            for fgrp, substruct in self.functional_group_dict.items()}

global i_start
i_start = 0

def gen_mashup(seed_mol, partner_mol, filters=None, filter_lipinski=False):
    #Have_Rad = False
    global i_start
    seed_s = seed_mol.pieces
    partner_s = partner_mol.pieces
    seed_w = frg_weight(seed_s)
    frag_w = frg_weight(partner_s)
    fragms = [Chem.MolFromSmiles(x) for x in partner_s]
    seedms = [Chem.MolFromSmiles(x) for x in seed_s]
    fragms = returnwmolist(frag_w,fragms)
    seedms = returnwmolist(seed_w,seedms)
    ms = BRICS.BRICSBuild(fragms, scrambleReagents=True, seeds=seedms)
    ms = list(ms)

    gen_mol_s = []
    for mol in ms:
        Chem.SanitizeMol(mol)
        if not filters == None:
            check_catalog = check_catalog_filters(mol, filters)
            if check_catalog:
                continue
        if filter_lipinski:
            check_lipinski = check_lipinski_filter(mol)
            if check_lipinski:
                continue
        gen_mol_s.append(mol)

    return gen_mol_s

def gen_fr_mutation(seed_mol, building_block_pool, filters=None, filter_lipinski=False):
    global i_start

    seed_s = seed_mol.pieces

    seed_w = frg_weight(seed_s)
    build_block_s = set()
    a = random.sample(building_block_pool, k=100)
    for i in a:
        build_block_s.update(i.pieces)

    seedms = [Chem.MolFromSmiles(x) for x in seed_s]
    buildms = [Chem.MolFromSmiles(x) for x in build_block_s]
    ms = BRICS.BRICSBuild(buildms, scrambleReagents=True, seeds=seedms)
    ms = list(ms)

    gen_mol_s = []

    for mol in ms:
        Chem.SanitizeMol(mol)
        if filters is not None:
            check_catalog = check_catalog_filters(mol, filters)
            if check_catalog:
                continue
        if filter_lipinski:
            check_lipinski = check_lipinski_filter(mol)
            if check_lipinski:
                continue

        gen_mol_s.append(mol)

    return gen_mol_s

def calc_tanimoto_distance(mol1, mol2):
    fp1 = AllChem.GetMorganFingerprint(mol1.RDKmol, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2.RDKmol, 2)
    tani = DataStructs.TanimotoSimilarity(fp1, fp2)
    dist = 1.0-tani
    return dist


def frg_weight(molpieces,disc=False):
    #b = []
    fragments = []
    molpiecelist = list(molpieces)
    molpiecelist = list(filter(None, molpiecelist))
    print(f"bf:{molpiecelist}")
    if len(molpiecelist) < 2:
        return np.array([1])
    for mol in molpiecelist:
        try:
            fragments.append(Molecule.from_smiles(mol, source="INITIAL"))
        except:
            print(mol)
            raise ValueError
    #fragments = [Molecule.from_smiles(mol, source="INITIAL") for mol in molpiecelist]
    frgs = [mol.RDKmol for mol in fragments]
    value = calcfrscore(frgs)

    values = np.array(value)
    if disc:
        sortedfrg_index = np.sort(values)
        delrange = round(len(frgs)*0.3)
        a = sortedfrg_index[0:delrange+1]
        print(delrange)
        molpiecelists = [molpiecelist[value.index(i)] for i in list(a)]
        print(f'mol:{molpiecelists}')

    else:
        # remove min value in fragments group

        weight = []
        # choose n-1 fragments from fragments group by weight from fragment score in synthetic arduousness score
        for n, (piece,i) in enumerate(zip(molpiecelist,value)):
            MoL = Chem.MolFromSmiles(piece)
            natoms = MoL.GetNumAtoms()
            i = i - (natoms**(1.005)-natoms)
            weight.append(10**i)

    return np.array(weight)


def check_hav_minus4(molecule):
    bi = {}
    num_min4 = 0
    mol = Chem.MolFromSmiles(molecule)
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2, bitInfo=bi)
    fps = fp.GetNonzeroElements()
    for x in fps:
        for i in range(len(bi[x])):
            score1 = fscores.get(x, -4)
            if score1 == -4:
                num_min4 += 1
                if num_min4 == 2:
                    return True
    return False


def fix_smiles(smiles):
    smiles = smiles.replace('[NH3]', '[NH3+]')
    smiles = smiles.replace('[NH2]', '[NH2+]')
    smiles = smiles.replace('[NH]', '[NH+]')
    return smiles
