import os
import json
import tempfile
import subprocess as sp
from typing import List

from rdkit import Chem
from rdkit.Chem import Recap,BRICS,Descriptors
from rdkit.Chem.Descriptors import NumRadicalElectrons
from openbabel import pybel
from openbabel.pybel import readfile, readstring

from libfilter import check_catalog_filters, check_lipinski_filter

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
            self.read_molecules_from_smiles_fn(input_fn)

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
        print(fragms)
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
    functional_group_dict: dict = get_dict_from_json_file(fn_func_json)

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
        with tempfile.NamedTemporaryFile(mode='w+t') as temp_fn_smi:
            temp_fn_smi.write(self.smiles)
            temp_fn_smi.flush()

            ret = sp.run(
                ["corina", '-i', "t=smiles", temp_fn_smi.name, "-o", "t=mol2"],
                stdout=sp.PIPE, check=True, text=True)

        os.unlink("corina.trc")

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
        # for BRICS retrosynthesis
        if method == 'BRICS':
            self.pieces = BRICS.BRICSDecompose(
                self.RDKmol, minFragmentSize=4,
                singlePass=True, keepNonLeafNodes=True)
        elif method == 'RECAP':
            self.pieces = Recap.RecapDecompose(self.RDKmol)

    def determine_functional_groups(self):
        # for in-silico reaction
        self.HasFunctionalGroup = {}
        for fgrp, smarts in self.functional_group_dict.items():
            substructure = Chem.MolFromSmarts(smarts)
            self.HasFunctionalGroup[fgrp] = \
                self.RDKmol.HasSubstructMatch(substructure)

    def draw(self, output_fn='output.png'):
        # 2d visualization of molecules
        mol_pybel = readstring('smi',self.smiles)
        mol_pybel.draw(show=False, filename=output_fn)


global i_start
i_start = 0


def gen_crossover(seed_mol, partner_mol, filters=None, filter_lipinski=False):
    Have_Rad = False
    global i_start
    seed_s = set(seed_mol.pieces)
    frag_s = set(partner_mol.pieces)

    fragms = [Chem.MolFromSmiles(x) for x in frag_s]
    seedms = [Chem.MolFromSmiles(x) for x in seed_s]
    #ms = BRICS.BRICSBuild(fragms, scrambleReagents=True, seeds=seed_s)
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
        if NumRadicalElectrons(mol) != 0:
            Have_Rad = True
        gen_mol_s.append(mol)

    return gen_mol_s, Have_Rad

def calc_tanimoto_distance(mol1, mol2):
    mol1 = str(mol1)
    mol2 = str(mol2)
    if mol1.find("~") != -1:
        mol1 = mol1.replace("~","")
    if mol2.find("~") != -1:
        mol2 = mol2.replace("~","")
    mol1 = pybel.readstring("smi",mol1)
    mol2 = pybel.readstring("smi",mol2)
    fp1 = mol1.calcfp(fptype="fp4")
    fp2 = mol2.calcfp(fptype="fp4")
    #mol1 = Chem.rdmolfiles.MolFromSmiles(str(mol1))
    #mol2 = Chem.rdmolfiles.MolFromSmiles(str(mol2))
    #fp1 = FingerprintMols.FingerprintMol(mol1)
    #fp2 = FingerprintMols.FingerprintMol(mol2)
    tani = fp1|fp2
    #tani = DataStructs.FingerprintSimilarity(fp1,fp2)
    dist = 1.0-tani
    return dist

def fix_smiles(smiles):
    smiles = smiles.replace('[NH3]', '[NH3+]')
    smiles = smiles.replace('[NH2]', '[NH2+]')
    smiles = smiles.replace('[NH]', '[NH+]')
    return smiles

if __name__=='__main__':
    mol_pool = Molecule_Pool('../example/radicalparents.smi')
#    mol_pool.gen_crossover()
#    #print mol_pool
#
#    fn_func_json = '/home/neclasic/Screen/GalaxyMolOpt/data/reaction_libraries/all_rxns/All_Rxns_functional_groups.json'
#    functional_group_dict = get_dict_from_json_file(fn_func_json)
#    #print functional_group_dict['Alcohol_clickchem']
#    mol_pool[0].determine_functional_groups(functional_group_dict)
#    print(mol_pool[0])
    #print(mol_pool)
    for i in range(0,len(mol_pool)):
        mol_pool[i].decompose()
        print(mol_pool[i].pieces)
    #gen_crossover(mol_pool[0], mol_pool[1])