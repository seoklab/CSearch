import random
import os
import json
import tempfile
import subprocess as sp
import numpy as np
from typing import List, Set

from rdkit import Chem
from rdkit.Chem import Recap, BRICS, AllChem, DataStructs
from rdkit.Chem.Descriptors import NumRadicalElectrons
from openbabel import pybel
from openbabel.pybel import readfile, readstring

from .libfilter import check_catalog_filters, check_lipinski_filter

__all__ = ["EXEC_CORINA", "Molecule_Pool", "Molecule",
           "get_dict_from_json_file", "gen_crossover",
           "calc_tanimoto_distance", "fix_smiles"]
EXEC_CORINA = '/applic/corina/corina'


def get_dict_from_json_file(fn_json):
    with open(fn_json, 'r') as fp:
        json_dict = json.loads(fp.read())
    return json_dict


class Molecule_Pool(object):
    # CSA bank related variables and functions
    def __init__(self, input_fn, n_mol=None):
        self.mol_s: List[Molecule] = []
        if input_fn.endswith('mol2'):
            self.read_molecules_from_mol2_fn_pybel(input_fn, n_mol)
        else:
            self.read_molecules_from_smiles_fn(input_fn, n_mol)

    def __len__(self):
        return len(self.mol_s)

    def __getitem__(self,i):
        return self.mol_s[i]

    def __repr__(self):
        smi_s = '\n'.join(['%s'%mol.smiles for mol in self.mol_s])
        return smi_s

    def read_molecules_from_smiles_fn(self, smi_fn):
        suppl = Chem.SmilesMolSupplier(smi_fn, delimiter='\t', titleLine=False)
        for RDKmol in suppl:
            mol = Molecule.from_rdkit(RDKmol, build_3d=True)
            self.mol_s.append(mol)

    def read_molecules_from_mol2_fn(self, mol2_fn, sanitize=True):
        # Giving many errors, kukelize problem
        mol2_block_s = []
        mol_lines = []
        i_mol = 0

        with open(mol2_fn, 'r') as f:
            for line in f:
                if line.startswith('@<TRIPOS>MOLECULE'):
                    if not i_mol == 0:
                        mol2_block = ''.join(mol_lines)
                        mol2_block_s.append(mol2_block)
                    mol_lines = []
                    i_mol += 1
                mol_lines.append(line)
            if i_mol > 0:
                mol2_block = ''.join(mol_lines)
                mol2_block_s.append(mol2_block)

        for block in mol2_block_s:
            RDKmol = Chem.MolFromMol2Block(block,sanitize=sanitize)
            mol = Molecule.from_rdkit(RDKmol)
            self.mol_s.append(mol)

    def read_molecules_from_mol2_fn_pybel(self, mol2_fn):
        for mol_pybel in readfile("mol2", mol2_fn):
            smiles = mol_pybel.write()
            smiles = fix_smiles(smiles) #OKAY?

            mol = Molecule.from_smiles(smiles, build_3d=True)
            self.mol_s.append(mol)

    def gen_crossover(self):
        frag_s = set()

        for i, mol in enumerate(self.mol_s):
            mol.decompose(method='BRICS')
            frag_s.update(mol.pieces)

            mol_block = Chem.MolToMolBlock(mol.RDKmol)
            mol_pybel = readstring('mol', mol_block)
            mol_pybel.draw(show=False, filename=f'generated/start_{i}.png')

        fragms = list(map(Chem.MolFromSmiles, frag_s))
        ms = BRICS.BRICSBuild(fragms, scrambleReagents=True)
        for i, mol in enumerate(ms):
            mol_block = Chem.MolToMolBlock(mol)
            mol_pybel = readstring('mol', mol_block)
            mol_pybel.draw(show=False, filename=f'generated/gen_{i}.png')

    def gen_fr_mutation(self):
        frag_s = set()

        for i, mol in enumerate(self.mol_s):
            mol.decompose(method='BRICS')
            frag_s.update(mol.pieces)

            mol_block = Chem.MolToMolBlock(mol.RDKmol)
            mol_pybel = readstring('mol', mol_block)
            mol_pybel.draw(show=False, filename=f'generated/start_{i}.png')

        fragms = list(map(Chem.MolFromSmiles, frag_s))
        ms = BRICS.BRICSBuild(fragms, scrambleReagents=True)
        for i, mol in enumerate(ms):
            mol_block = Chem.MolToMolBlock(mol)
            mol_pybel = readstring('mol', mol_block)
            mol_pybel.draw(show=False, filename=f'generated/gen_{i}.png')


    def determine_functional_groups(self):
        for mol in self.mol_s:
            if not len(mol.HasFunctionalGroup) == 0:
                continue
            mol.determine_functional_groups()


class Molecule(object):
    fn_func_json = '/home/hakjean/galaxy2/developments/MolGen/db_chembl/All_Rxns_functional_groups.json'
    functional_group_dict: dict = {
        fgrp: Chem.MolFromSmarts(smarts)
        for fgrp, smarts in get_dict_from_json_file(fn_func_json).items()}

    ##MOVE MOLECULE TO CORE
    # read molecule information
    def __init__(self, smiles, RDKmol, source=None, build_3d=False):
        self.smiles = smiles
        self.RDKmol = RDKmol
        self.source = source

        self.HasFunctionalGroup = {}
        self.determine_functional_groups()

        self.is_3d: bool = build_3d
        self.mol2_block: List[str] = []
        if build_3d:
            self._build_mol2_3D()

        self.pieces: Set[str] = set()

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

    def decompose(self, method='BRICS'):
        if self.pieces:
            return

        # for BRICS retrosynthesis
        if method == 'BRICS':
            pieces = BRICS.BRICSDecompose(
                self.RDKmol, minFragmentSize=4,
                singlePass=True, keepNonLeafNodes=True)
        elif method == 'RECAP':
            pieces = Recap.RecapDecompose(self.RDKmol)
        else:
            raise NotImplementedError

        self.pieces = set(pieces)

    def determine_functional_groups(self):
        # for in-silico reaction
        self.HasFunctionalGroup = {
            fgrp: self.RDKmol.HasSubstructMatch(substruct)
            for fgrp, substruct in self.functional_group_dict.items()}

    def draw(self, output_fn='output.png'):
        # 2d visualization of molecules
        mol_pybel = readstring('smi',self.smiles)
        mol_pybel.draw(show=False, filename=output_fn)


global i_start
i_start = 0


def gen_crossover(seed_mol, partner_mol, filters=None, filter_lipinski=False):
    Have_Rad = False
    global i_start
    seed_s = seed_mol.pieces
    frag_s = partner_mol.pieces

    fragms = [Chem.MolFromSmiles(x) for x in frag_s]
    seedms = [Chem.MolFromSmiles(x) for x in seed_s]
    #ms = BRICS.BRICSBuild(fragms, scrambleReagents=True, seeds=seed_s)
    ms = BRICS.BRICSBuild(fragms, scrambleReagents=True, seeds=seedms)
    ms = list(ms)

    gen_mol_s = []
    rad_mol_s = []
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
        if NumRadicalElectrons(mol) != 0:
            if '[NH' or '[P]' in str(Chem.MolToSmiles(mol)):
                continue
            Have_Rad = True
            rad_mol_s.append(mol)
            continue
        gen_mol_s.append(mol)

    return gen_mol_s, rad_mol_s, Have_Rad

def gen_fr_mutation(seed_mol, building_block_pool, filters=None, filter_lipinski=False):
    global i_start
    Have_Rad = False
    seed_s = seed_mol.pieces

    build_block_s = set()
    a = random.sample(building_block_pool, k=50)
    for i in a:
        build_block_s.update(i.pieces)

    seedms = [Chem.MolFromSmiles(x) for x in seed_s]
    buildms = [Chem.MolFromSmiles(x) for x in build_block_s]
    ms = BRICS.BRICSBuild(buildms, scrambleReagents=True, seeds=seedms)
    ms = list(ms)

    gen_mol_s = []
    rad_mol_s = []
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
        if NumRadicalElectrons(mol) != 0:
            if '[NH' or '[P]' in str(Chem.MolToSmiles(mol)):
                continue
            Have_Rad = True
            rad_mol_s.append(mol)
            continue
        gen_mol_s.append(mol)

    return gen_mol_s, rad_mol_s, Have_Rad


def calc_tanimoto_distance(mol1, mol2):
    fp1 = AllChem.GetMorganFingerprint(mol1.RDKmol, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2.RDKmol, 2)
    tani = DataStructs.TanimotoSimilarity(fp1, fp2)
    dist = 1.0-tani
    return dist

def fix_smiles(smiles):
    smiles = smiles.replace('[NH3]', '[NH3+]')
    smiles = smiles.replace('[NH2]', '[NH2+]')
    smiles = smiles.replace('[NH]', '[NH+]')
    return smiles

if __name__=='__main__':
    mol_pool = Molecule_Pool('sample.mol2')
#    mol_pool.gen_crossover()
#    #print mol_pool
#
#    fn_func_json = '/home/neclasic/Screen/GalaxyMolOpt/data/reaction_libraries/all_rxns/All_Rxns_functional_groups.json'
#    functional_group_dict = get_dict_from_json_file(fn_func_json)
#    #print functional_group_dict['Alcohol_clickchem']
#    mol_pool[0].determine_functional_groups(functional_group_dict)
#    print(mol_pool[0])

    mol_pool[0].decompose()
    mol_pool[1].decompose()
    gen_crossover(mol_pool[0], mol_pool[1])
