import os
import glob
import random
from typing import List

import Galaxy
import numpy as np
from scipy.spatial import distance as D
from rdkit import Chem

from libfilter import prepare_catalog_filters
from libfilter import check_lipinski_filter
from opps.fragment_merge import *
from opps.in_silico_reaction import (
    Reaction, get_dict_from_json_file, get_compl_mol_dict)
from opps.similarity_search import read_database_mol
from opps.energy_calculation_tab import energy_calc


N_PROC_DOCK = 1


class CSA(object):
    def __init__(self, filter_lipinski=False, use_ML=True, ref_lig=None):
        #self.n_bank=192
        #self.n_bank_add=50

        #self.n_seed = 48
        self.n_bank = 48
        self.n_bank_add = 50
        self.n_seed = 24
        #'dcut1': 2.0, # initial Dcut is the initial average diff / dcut1
        #'dcut2': 5.0, # minimum Dcut is the initial average diff / dcut2
        self.seed_mask = []  # any index in the list is not selected as a seed
        self.n_csa_iter = 1
        self.n_seed_cycle = 1
        self.max_opt_cycle = 10000
        self.dist_mat = np.zeros((self.n_bank, self.n_bank))
        self.use_ML = use_ML

        self.seed_cycle = 1

        self.n_opt_to_D_min = 10000

        fn_database_mol = '/home/hakjean/galaxy2/developments/MolGen/db_chembl/2020-01-BioDesign.sdf'

        self.catalog_filters = prepare_catalog_filters(PAINS=True)
        self.filter_lipinski=filter_lipinski
        #if filter_PAINS.HasMatch(mol):
        #    continue

        #prep similarity search
        self.database_mol_s, self.database_mol_fp_s = read_database_mol(fn_database_mol)

        ### TEMP, load model
        #with torch.no_grad():
        #    # ML model load
        #    model_s = [MyModel().to(device) for _ in MODEL_FN_S]
        #    [model.load_state_dict(torch.load(fn, map_location=device)['model_state_dict']) for model,fn in zip(model_s,MODEL_FN_S)]
        #    [model.eval() for model in model_s]
        #self.model_s = model_s

        self.ref_lig = ref_lig

    def initialize_csa(self, job, init_bank_smiles_fn, n_proc=None):
        #self.prot_fn = prot_fn
        #self.cntr_R = cntr_R
        self.job = job
        n_proc = Galaxy.core.define_n_proc(n_proc)
        n_proc=24
        self.n_proc = n_proc

        #initialize in-silico reactions
        fn_rxn_lib = '/home/hakjean/galaxy2/developments/MolGen/db_chembl/All_Rxns_rxn_library.json'
        reaction_dict = get_dict_from_json_file(fn_rxn_lib)
        fn_compl_mol_s = glob.glob('/home/hakjean/galaxy2/developments/MolGen/db_chembl/complementary_mol_dir/*.smi')
        self.compl_mol_dict = get_compl_mol_dict(fn_compl_mol_s)
        self.reaction_s: List[Reaction] = []
        for reaction_name in reaction_dict:
            self.reaction_s.append(Reaction(reaction_dict[reaction_name]))

        #initial bank
        self.read_initial_bank(init_bank_smiles_fn)
        self.setup_initial_Dcut()

    def run(self):
        self.write_bank(0)
        for i_csa_iter in range(self.n_csa_iter):
            for i_opt_cycle in range(1,self.max_opt_cycle+1):
                self.select_seeds()
                new_smiles_s, new_energy_s = self.make_new_confs(i_opt_cycle)
                print(len(new_smiles_s))
                #print(type(new_smiles_s))
                print(len(new_energy_s))
                #print(type(new_energy_s))
                #print(new_smiles_s)
                #print(new_energy_s)
                self.update_bank(new_smiles_s, new_energy_s)
                self.update_distance()
                self.update_functional_groups()
                self.print_log()
                print('CSA Run Number / Iteration / Seed cylce = %d / %d / %d'%(i_csa_iter, i_opt_cycle, \
                        self.seed_cycle))
                print('D_avg/D_cut/D_min=%.3f/%.3f/%.3f'%(self.Davg, self.Dcut, self.Dcut_min))
                self.write_bank(i_opt_cycle)
                #update_distance
                self.D_cut = self.D_cut * self.xctdif
                self.D_cut = max(self.D_cut, self.D_min)
                print(f"D_cut : {self.D_cut}")
                if len(self.seed_mask) ==0:
                    return

    def read_initial_bank(self, smiles_fn):
        self.bank_pool: List[Molecule] = []

        with open(smiles_fn, 'r') as f_n:
            for i_mol, line in enumerate(f_n):
                smile_str = line.split()
                smiles_str = smile_str[0]
                smiles = smiles_str.replace('[NH3]', '[NH3+]')
                smiles = smiles.replace('[NH2]', '[NH2+]')
                smiles = smiles.replace('[NH]', '[NH+]')

                mol = Molecule.from_smiles(smiles, source="INITIAL")
                if mol.RDKmol is None:
                    print(f'error processing {i_mol}th molecule')
                    continue
                mol.decompose()
                self.bank_pool.append(mol)
                if len(self.bank_pool) == self.n_bank:
                    break

        self.update_functional_groups()
        self.energy_bank_pool = energy_calc(
            [mol.smiles.replace("~", "") for mol in self.bank_pool], "csa")

        print("initial bank energy:")
        print(self.energy_bank_pool)
        self.job.mkdir('cycle_0')
        smiles_block_s = []
        with open(smiles_fn, 'r') as fp:
            for line in fp:
                smiles_block_s.append(line)
        #mol2_block_s = []
        #with open(mol2_fn.relpath(), 'r') as fp:
        #    mol2_lines = fp.readlines()
#
        # split mol2 fn
#        mol2_block = []
#        for line in mol2_lines:
#            if line.startswith('#'):
#                continue
#            if line.startswith('@<TRIPOS>MOLECULE'):
#                if not len(mol2_block) == 0:
#                    mol2_block_s.append(mol2_block)
#                    mol2_block = []
#            mol2_block.append(line)
#        mol2_block_s.append(mol2_block)

        self.job.chdir_prev()

        #store initial bank
        self.init_bank = self.bank_pool

    @staticmethod
    def _tanimoto_dist(m, n):
        return calc_tanimoto_distance(m[0], n[0])

    @classmethod
    def _cdist_mols(cls, mols: List[Molecule]):
        mols_arr = np.array(mols, dtype=object)[:, np.newaxis]
        return D.cdist(mols_arr, mols_arr, metric=cls._tanimoto_dist)

    def update_distance(self):
        self.dist_mat = self._cdist_mols(self.bank_pool)

    def setup_initial_Dcut(self):
        self.update_distance()
        self.Davg = np.average(self.dist_mat)

        self.Dcut = self.Davg / 2.0  # factor_init_D_cut
        self.Dcut_min = self.Davg / 5.0 # factor_min_D_cut

        self.D_min = self.Davg / 5.0

        nst = float(self.n_opt_to_D_min) / len(self.bank_pool)
        self.xctdif = (self.D_cut / self.D_min)**(-1.0 / nst)

    def select_seeds(self):
        print(f"seed_mask : {self.seed_mask}")
        n_unused = len(self.seed_mask)
        if n_unused > self.n_seed:
            n_seed_from_unused = self.n_seed
            n_seed_from_used = 0
        else:
            n_seed_from_unused = n_unused
            n_seed_from_used = self.n_seed - n_unused

        #pick from unused
        seed_selected_unused = random.sample(self.seed_mask, n_seed_from_unused)
        #store used seed
        for i_seed in seed_selected_unused:
            self.seed_mask.remove(i_seed)

        #pick rest from used
        used_mask = [i for i in range(1,self.n_seed+1) if not i in self.seed_mask]
        seed_selected_used = random.sample(used_mask, n_seed_from_used)

        seed_selected = seed_selected_unused + seed_selected_used
        #print(f"seed_selected:"seed_selected)
        return seed_selected

    def update_functional_groups(self):
        for mol in self.bank_pool:
            mol.determine_functional_groups()

    def make_new_confs(self, i_cycle, max_gen: int = 48):
        seed_selected = self.select_seeds()
        #print(seed_selected)
        #print(str(seed_selected))
        new_mol_s: List[Molecule] = []
        # generate molecules
        max_select_brics = 5
        # crossover, BRICS

        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            #seed_mol = self.init_bank[0]
            partner_mol = random.choice(self.init_bank)

            gen_RDKmol_s = gen_crossover(seed_mol, partner_mol,
                                         filters=self.catalog_filters,
                                         filter_lipinski=self.filter_lipinski)

            if len(gen_RDKmol_s) > max_select_brics:
                gen_RDKmol_s = random.sample(gen_RDKmol_s, max_select_brics)

            for gen_RDKmol in gen_RDKmol_s:
                try:
                    new_mol = Molecule.from_rdkit(
                        gen_RDKmol, build_3d=True, source='BRICS')
                    new_mol_s.append(new_mol)
                    if len(new_mol_s) >= max_gen:
                        break
                except Exception:
                    continue

            if len(new_mol_s) >= max_gen:
                break

        max_select_reaction = 3
        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            random.shuffle(self.reaction_s)

            products_all: List[Molecule] = []
            finish_reaction = False

            for reaction in self.reaction_s:
                groups_found, _ = reaction.check_reaction_components(seed_mol)
                if len(groups_found) == 0:
                    continue
                group_picked = random.choice(groups_found)

                reactants = {}
                reactants[group_picked] = seed_mol
                for group in reaction.functional_groups:
                    if group_picked == group:
                        continue
                    compl_mol = random.choice(self.compl_mol_dict[group])
                    reactants[group] = compl_mol

                products = reaction.run_reaction(
                    reactants,
                    filters=self.catalog_filters,
                    filter_lipinski=self.filter_lipinski,
                    build_3d=True)
                random.shuffle(products)

                for product in products:
                    products_all.append(product)
                    if len(products_all) >= max_select_reaction:
                        finish_reaction =True
                        break

                if finish_reaction:
                    break

            new_mol_s += products_all
        ## similarity search
        #for i_seed in seed_selected:
        #    seed_mol = self.bank_pool[i_seed]
        #    gen_mol_s = pick_similar_compound(seed_mol, self.database_mol_s, self.database_mol_fp_s, \
        #            filters=self.catalog_filters, filter_lipinski=self.filter_lipinski)
        #    for new_mol in gen_mol_s:
        #        new_mol.source='DB'
        #    new_mol_s += gen_mol_s


        # calc energy
       # for i_seed in new_mol_s:
            #seed_mol = self.bank_pool[i_seed]
            #print('seed')
            #print()
        #    seed_mol = str(i_seed)
         #   energy_s = energy_calc(seed_mol,"csa")
            #print('New_mol :')
            #print(new_mol_s)

          #  new_energy_s.append(energy_s)

        new_energy_s = energy_calc(
            [mol.smiles for mol in new_mol_s], "csa")
        #new_energy_s = energy_calc(new_mol_s, multi=True)
        #print(new_energy_s[0])
        return new_mol_s, new_energy_s

    def update_bank(
            self, new_mol_s: List[Molecule], new_energy_s: List[float]):
        for i, (i_mol, i_energy) in enumerate(zip(new_mol_s, new_energy_s)):
            i_Emax_bank_u = np.argmax(self.energy_bank_pool)

            if i_energy >= self.energy_bank_pool[i_Emax_bank_u]:
                continue

            #i_mol = new_mol_s[i_new]
            # check lipinski filter
            smi_mol = Chem.MolFromSmiles(i_mol.smiles)
            if check_lipinski_filter(smi_mol):
                continue
            # check closest bank member

            dist_s = [calc_tanimoto_distance(bank_mol, i_mol)
                      for bank_mol in self.bank_pool]
            min_idx = np.argmin(dist_s)
            min_dist = dist_s[min_idx]
            i_mol.RDKmol.UpdatePropertyCache()
            Chem.GetSymmSSSR(i_mol.RDKmol)
            i_mol.decompose() ######## move to somewhere

            #replace current bank
            if (min_dist < self.Dcut):
                if i_energy < new_energy_s[min_idx]:
                    print('%d %.3f was replaced to %d %.3f in same group'
                          % (min_idx, new_energy_s[min_idx], i, i_energy))

                    print(f"before {self.bank_pool[min_idx]} after {i_mol}")
                    self.bank_pool[min_idx] = i_mol
                    self.energy_bank_pool[min_idx] = i_energy
                    #new_energy_s[min_idx] = i_energy
                    if not min_idx in self.seed_mask:
                        self.seed_mask.append(min_idx)
            else:
                print(
                    '%d %.3f was replaced to %d %.3f in new group'
                    %(i_Emax_bank_u, new_energy_s[i_Emax_bank_u], i, i_energy))

                print(f"before {self.bank_pool[i_Emax_bank_u]} after {i_mol}")
                self.bank_pool[i_Emax_bank_u] = i_mol
                self.energy_bank_pool[i_Emax_bank_u] = i_energy
                #new_energy_s[i_Emax_bank_u] = i_energy
                if not i_Emax_bank_u in self.seed_mask:
                    self.seed_mask.append(i_Emax_bank_u)

    def print_log(self):
        for i in range(self.n_bank):
            print('Bank %d:success'%(i+1))

    def write_bank(self, i_cycle):
        with open('single/csa_%d.log'%i_cycle, 'wt') as fp:
            for i_mol, mol in enumerate(self.bank_pool):
                print(i_mol)
                print(mol)
                mol_energy = energy_calc(mol.smiles, 'single')
                fp.write(f'{i_mol+1} {float(mol_energy) * 100}\n')
                fp.write(f'{mol}\n')


if __name__=='__main__':
    import sys
    #prot_fn = Galaxy.core.FilePath(sys.argv[1])
#        smiles_fn = Galaxy.core.FilePath(sys.argv[2])
    smiles_fn = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA/data/initial_bank_02.smi'
    smiles_fn = os.path.abspath(smiles_fn)
    #smiles_fn = os.path.relpat
    #opt_fn =  sys.argv[3]

    #cntr = [0.0,0.0,0.0]
    #with open(opt_fn, 'r') as fp:
    #    lines = fp.readlines()

    #for line in lines:
    #    linesp=line.split('=')
    #    if linesp[0] == 'X':
    #        cntr[0] = float(linesp[1])
    #    if linesp[0] == 'Y':
    #        cntr[1] = float(linesp[1])
    #    if linesp[0] == 'Z':
    #        cntr[2] = float(linesp[1])


    job = Galaxy.initialize(title='Test_OptMol', mkdir=False)
    #cntr  = tuple(cntr)
    csa = CSA()
    csa.initialize_csa(job, smiles_fn)
    csa.run()
