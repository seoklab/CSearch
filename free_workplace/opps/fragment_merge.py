import sys
import os
from openbabel import openbabel
from openbabel import pybel
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Recap,BRICS
from rdkit.Chem import rdmolfiles
from rdkit.Chem.Fingerprints import FingerprintMols
import tempfile
from libfilter import check_catalog_filters, check_lipinski_filter
from energy_calculation import energy_calc
sys.path.append('/home/hakjean/galaxy2/openbabel/openbabel-2.3.1/scripts/python')
from pybel import *

EXEC_CORINA = '/applic/corina/corina'

class Molecule_Pool(object):
    # CSA bank related variables and functions
    def __init__(self, input_fn, n_mol=None):
        self.mol_s = []
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
    def read_molecules_from_smiles_fn(self,smi_fn):
        suppl = Chem.SmilesMolSupplier(smi_fn,delimiter='\t',titleLine=False)
        for RDKmol in suppl:
            mol = Molecule(RDKmol=RDKmol, build_3d=True)
    def read_molecules_from_mol2_fn(self,mol2_fn, sanitize=True):
        # Giving many errors, kukelize problem
        mol2_block_s = []
        mol_lines = []
        i_mol = 0
        with open(mol2_fn, 'r') as f:
            mol2_lines = f.readlines()
            for line in mol2_lines:
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
            mol = Molecule(RDKmol=RDKmol)
            self.mol_s.append(mol)

    def read_molecules_from_mol2_fn_pybel(self,mol2_fn, n_mol):
        for mol_pybel in readfile("mol2", mol2_fn):
            smiles = mol_pybel.write()
            smiles=fix_smiles(smiles) #OKAY?
            mol = Molecule(smiles=smiles, build_3d=True)
            self.mol_s.append(mol)

    def gen_crossover(self):
        frag_s = set()

        i_mol =0
        for mol in self.mol_s:
            mol.decompose(method='BRICS')
            frag_s.update(mol.pieces)

            mol_block = Chem.MolToMolBlock(mol.RDKmol)
            mol_pybel = readstring('mol',mol_block)
            i_mol += 1
            mol_pybel.draw(show=False,filename = 'generated/start_%d.png'%i_mol)

        fragms = [Chem.MolFromSmiles(x) for x in frag_s]

        ms = BRICS.BRICSBuild(fragms, scrambleReagents=True)
        #print(ms.next())

        i_mol = 0
        for mol in ms:
            i_mol += 1
            mol_block = Chem.MolToMolBlock(mol)
            mol_pybel = readstring('mol',mol_block)
            smiles=mol_pybel.write()
            mol_pybel.draw(show=False,filename = 'generated/gen_%d.png'%i_mol)
    def determine_functional_groups(self, functional_group_dict):
        for mol in self.mol_s:
            if not len(mol.HasFunctionalGroup) == 0:
                continue
            mol.determine_functional_groups(functional_group_dict)

class Molecule(object):
    ##MOVE MOLECULE TO CORE
    # read molecule information
    def __init__(self, smiles=None, RDKmol=None, mol2_fn=None, build_3d=False):
        self.mol2_block = []
        self.HasFunctionalGroup = {}
        if not RDKmol==None:
            self.RDKmol = RDKmol
            self.read_smiles()
        if not smiles==None:
            self.read_molecule_from_smiles(smiles)
        if build_3d:
            self.build_mol2_3D()

#        self.pybelmol = pybel.readstring('smi',self.smiles)

    def __repr__(self):
        return self.smiles
    def read_smiles(self):
        self.smiles = Chem.MolToSmiles(self.RDKmol)
    def read_molecule_from_smiles(self, smiles):
        self.smiles = smiles
        self.RDKmol = Chem.MolFromSmiles(self.smiles)
    def read_molecule(self):
        self.RDKmol = Chem.MolFromSmiles(self.smiles)
#    def read_molecules_from_mol2_fn_pybel(self,mol2_fn):
#        for mol_pybel in pybel.readfile("mol2", mol2_fn):
#            smiles = mol_pybel.write()
#            mol = Molecule(smiles=smiles, build_3d=True)
#            self.mol_s.append(mol)
    def energy_molecule(self):
        smiles_k = self.smiles
        self.galigandE = energy_calc(smiles_k,input_file=None, input_type='smiles')
    def build_mol2_3D(self, method='corina'):
        if method == 'corina':
            with tempfile.NamedTemporaryFile(mode='w+t') as temp_fn_smi:
                temp_fn_smi.write(self.smiles)
                temp_fn_smi.seek(0)
                cmd = '%s -i t=smiles %s -o t=mol2'%(EXEC_CORINA, temp_fn_smi.name)
                pipe = os.popen(cmd)
                self.mol2_block = pipe.readlines()
                pipe.close()
            self.mol2_block = [line for line in self.mol2_block if not line.startswith('#')]
            os.system('rm corina.trc')
    def decompose(self, method='BRICS'):
        # for BRICS retrosynthesis
        if method == 'BRICS':
            self.pieces = BRICS.BRICSDecompose(self.RDKmol, minFragmentSize=4, \
                                               singlePass=True, keepNonLeafNodes=True)
        elif method == 'RECAP':
            self.pieces = Recap.RecapDecompose(self.RDKmol)
    def determine_functional_groups(self, functional_group_dict):
        # for in-silico reaction
        self.HasFunctionalGroup = {}
        for functional_group in functional_group_dict:
            smarts = functional_group_dict[functional_group]
            substructure = Chem.MolFromSmarts(smarts)
            if self.RDKmol.HasSubstructMatch(substructure):
                self.HasFunctionalGroup[functional_group] = True
            else:
                self.HasFunctionalGroup[functional_group] = False
    def draw(self, output_fn='output.png'):
        # 2d visualization of molecules
        mol_pybel = readstring('smi',self.smiles)
        mol_pybel.draw(show=False,filename=output_fn)

def get_dict_from_json_file(fn_json):
    import json
    with open(fn_json,'r') as fp:
        json_dict = json.loads(fp.read())
    return json_dict

global i_start
i_start=0

def gen_crossover(seed_mol, partner_mol, filters=None, filter_lipinski=False):
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
        gen_mol_s.append(mol)

    return gen_mol_s

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
