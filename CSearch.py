from __future__ import print_function
import os
import random
import argparse
from datetime import date
from time import time
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import Galaxy
import numpy as np
from scipy.spatial import distance as D

from opps.libs.utils import str2bool
from opps.libfilter import prepare_catalog_filters, check_lipinski_filter
from opps.fragment_merge import (
    Molecule, gen_fr_mutation, gen_mashup, calc_tanimoto_distance)
from opps.energy_calculation import energy_calc, qed_calc, sa_calc



class CSA(object):
    def __init__(self, ref_lig=None):
        self.n_bank = args.banksize
        self.n_seed = args.seed_num
        self.nst = args.ann
        self.seed_mask = []  # any index in the list is not selected as a seed
        self.n_csa_iter = 1
        self.n_seed_cycle = args.seed_cycle
        self.max_opt_cycle = args.maxiter
        self.dist_mat = np.zeros((self.n_bank, self.n_bank))
        self.dcut_cycle = 1
        self.n_dcut_cycle = 2
        self.seed_cycle = 1
        self.pdbid = args.pdbid
        self.catalog_filters = prepare_catalog_filters(PAINS=True)
        self.filter_lipinski = args.filter
        self.ref_lig = ref_lig
        self.bankset = set([])
        self.date = date.today()
        if int(args.num) < 10:
            self.num = 0 + args.num
        else:
            self.num = args.num


    def initialize_csa(self, job, init_bank_smiles_fn,
                       building_blocks_smiles_fn, n_proc=None):
        self.job = job
        n_proc = Galaxy.core.define_n_proc(n_proc)
        n_proc = 8
        self.n_proc = n_proc
        self.type = init_bank_smiles_fn[-8:-4]
        # initial bank
        self.read_initial_bank(init_bank_smiles_fn)
        self.building_block_setup(building_blocks_smiles_fn)
        self.setup_initial_Dcut()

    def run(self):
        self.write_bank(0)
        totalop = 0
        for i_csa_iter in range(self.n_csa_iter):
            for i_opt_cycle in range(1, self.max_opt_cycle + 1):
                self.select_seeds()
                (new_smiles_s, new_energy_s, new_mol_gen_type, new_qed_s,
                 new_sa_s, operat_count) = self.make_new_confs()
                (a, b) = operat_count
                totalop = a + b + totalop
                print(f'Mutation operator : {b}, Mashup operator : {a}')
                self.update_bank(new_smiles_s, new_energy_s, new_mol_gen_type,
                                 new_qed_s, new_sa_s)
                self.bankset.update(self.bank_pool)
                self.update_distance()
                self.update_functional_groups()
                print('CSA Run Number / Iteration / Seed cycle = '
                      f'{i_csa_iter} / {i_opt_cycle} / {self.seed_cycle}')
                print(f'D_avg/D_cut/D_min={self.D_avg:.3f}/{self.D_cut:.3f}'
                      f'/{self.D_cut_min:.3f}')
                self.write_bank(i_opt_cycle)
                # update_distance
                self.D_cut = self.D_cut * self.xctdif
                self.D_cut = max(self.D_cut, self.D_min)
                print(f"D_cut : {self.D_cut}")
                #d_cut cycle update
                if self.D_cut < self.D_min:
                    if self.dcut_cycle >= self.n_dcut_cycle:
                        continue
                    self.dcut_cycle += 1
                    self.setup_initial_Dcut()
                if len(self.seed_mask) == 0:
                    if self.seed_cycle == self.n_seed_cycle:
                        print(f'Total op {totalop}')
                        return
                    self.seed_cycle += 1
                    self.select_seeds()
                if i_opt_cycle == self.max_opt_cycle:
                    print(f'Total op {totalop}') 


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
                self.bank_pool.append(mol)
                if len(self.bank_pool) == self.n_bank:
                    break
        print('str:{str(mol)}')
        self.update_functional_groups()
        initial_mols = [mol.RDKmol for mol in self.bank_pool]
        self.energy_bank_pool = energy_calc(initial_mols, "csa", self.pdbid)
        self.bank_gen_type = ['o'] * self.n_bank
        self.bank_qed_s = qed_calc(initial_mols)
        self.bank_sa_s = sa_calc(initial_mols)
        self.bank_frg = [i.pieces for i in self.bank_pool]
        self.out_dir = f"Result/{self.pdbid}/{self.date}/Bank{self.n_bank}_seed{self.n_seed}_sc{self.n_seed_cycle}_mx{self.max_opt_cycle}_nst{self.nst}_{self.type}_{args.dmin}_{self.num}"
        self.job.mkdir(self.out_dir, cd=False)
        try:
            os.system(f"rm {self.out_dir}/*")
        except:
            print('no file')

       # store initial bank
        self.init_bank = self.bank_pool
        self.bankset = set(self.bank_pool)

    def building_block_setup(self, smiles_fn):
        self.building_pool: List[Molecule] = []

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
                self.building_pool.append(mol)
                if len(self.building_pool) == 20 * self.n_bank:
                    break

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
        self.D_avg = np.average(self.dist_mat)
        dmin = args.dmin 
        self.D_cut = self.D_avg / 2.0  # factor_init_D_cut
        self.D_cut_min = self.D_avg / dmin   # factor_min_D_cut
        self.D_min = self.D_avg / dmin
        nst = self.nst
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

        # pick from unused
        seed_selected_unused = random.sample(
            self.seed_mask, n_seed_from_unused)
        # store used seed
        for i_seed in seed_selected_unused:
            self.seed_mask.remove(i_seed)

        # pick rest from used
        used_mask = [i for i in range(1, self.n_seed + 1) if i not in self.seed_mask]
        seed_selected_used = random.sample(used_mask, n_seed_from_used)

        seed_selected = seed_selected_unused + seed_selected_used
        print(f"seed_updated : {seed_selected}")
        return seed_selected

    def update_functional_groups(self):
        for mol in self.bank_pool:
            mol.determine_functional_groups()

    def make_new_confs(self):
        seed_selected = self.select_seeds()
        new_mol_gen_type = []
        new_mol_s: List[Molecule] = []
        mutation_all_s: List[Molecule] = []
        mutation_count = 0
        # generate molecules
        max_select_brics = 10
        # mashup, BRICS
        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            partner_i = random.randrange(0,self.n_bank)
            #get partner molecule from the initial bank
            partner_mol = self.init_bank[partner_i]
            print(self.catalog_filters)
            gen_RDKmol_s = gen_mashup(seed_mol, partner_mol, filters=self.catalog_filters, filter_lipinski=self.filter_lipinski)
            if len(gen_RDKmol_s) > max_select_brics:
                gen_RDKmol_s = random.sample(gen_RDKmol_s, max_select_brics)

            for gen_RDKmol in gen_RDKmol_s:
                try:
                    new_mol = Molecule.from_rdkit(gen_RDKmol, source='BRICS')
                    new_mol_s.append(new_mol)
                    new_mol_gen_type.append(f'fr_seed:{i_seed}_initial:{partner_i}')
                    if len(new_mol_s) >= self.n_bank:
                        break
                except Exception:
                    continue

            if len(new_mol_s) >= self.n_bank:
                break
        frag_merge_count = len(new_mol_s)
        # mutation,BRICS
        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            gen_RDKmol_s = gen_fr_mutation(
                seed_mol, self.building_pool,
                filters=self.catalog_filters,
                filter_lipinski=self.filter_lipinski)
            if len(gen_RDKmol_s) > max_select_brics:
                gen_RDKmol_s = random.sample(gen_RDKmol_s, max_select_brics)
              
            for gen_RDKmol in gen_RDKmol_s:
                try:
                    mutation_all = Molecule.from_rdkit(
                        gen_RDKmol, source='BRICS')
                    mutation_all_s.append(mutation_all)
                    new_mol_gen_type.append(f'mut_seed:{i_seed}')
                    if len(mutation_all_s) >= self.n_bank:
                        break
                except Exception:
                    continue

            if len(mutation_all_s) >= self.n_bank:
                break

        new_mol_s += mutation_all_s
        print(new_mol_gen_type)
        mutation_count += len(mutation_all_s)
        mols = [mol.RDKmol for mol in new_mol_s]
        
        new_qed_s = qed_calc(mols)
        new_sa_s = sa_calc(mols)
        new_energy_s = energy_calc(mols, "csa", self.pdbid)
        operat_count = (frag_merge_count, mutation_count)
        return (new_mol_s, new_energy_s, new_mol_gen_type,
                new_qed_s, new_sa_s, operat_count)

    def update_bank(self, new_mol_s: List[Molecule], new_energy_s: List[float],
                    new_mol_gen_type: List[str], new_qed_s: List[float],
                    new_sa_s: List[float]):
        for i, (i_mol, i_energy, i_gentype, i_qed_s, i_sa_s) in enumerate(
                zip(new_mol_s, new_energy_s, new_mol_gen_type,
                    new_qed_s, new_sa_s)):
            i_Emax_bank_u = np.argmax(self.energy_bank_pool)
            
            # check lipinski filter
            if check_lipinski_filter(i_mol.RDKmol):
                continue
            # check closest bank member

            dist_s = [calc_tanimoto_distance(bank_mol, i_mol)
                      for bank_mol in self.bank_pool]
            min_idx = np.argmin(dist_s)
            min_dist = dist_s[min_idx]
            # replace current bank
            if (min_dist < self.D_cut):
                if i_energy < self.energy_bank_pool[min_idx]:
                    print(f'B{min_idx} {self.energy_bank_pool[min_idx]:.3f} '
                          f'was replaced to {i} {i_energy:.3f} in same group')

                    print(f"before {self.bank_pool[min_idx]} after {i_mol}")
                    self.bank_pool[min_idx] = i_mol
                    self.energy_bank_pool[min_idx] = i_energy
                    self.bank_gen_type[min_idx] = i_gentype
                    self.bank_qed_s[min_idx] = i_qed_s
                    self.bank_sa_s[min_idx] = i_sa_s
                    if min_idx not in self.seed_mask:
                        self.seed_mask.append(min_idx)
            else:
                if i_energy >= self.energy_bank_pool[i_Emax_bank_u]:
                    continue
                print(f'B{i_Emax_bank_u} '
                      f'{self.energy_bank_pool[i_Emax_bank_u]:.3f} was '
                      f'replaced to {i} {i_energy:.3f} in new group')
                print(f"before {self.bank_pool[i_Emax_bank_u]} after {i_mol}")
                self.bank_pool[i_Emax_bank_u] = i_mol
                self.energy_bank_pool[i_Emax_bank_u] = i_energy
                self.bank_gen_type[i_Emax_bank_u] = i_gentype
                self.bank_qed_s[i_Emax_bank_u] = i_qed_s
                self.bank_sa_s[i_Emax_bank_u] = i_sa_s
                if i_Emax_bank_u not in self.seed_mask:
                    self.seed_mask.append(i_Emax_bank_u)



    def write_bank(self, i_cycle):
        first = True
        if self.pdbid == '6M0K':
            x = 10
        else:
            x = 100
        with open(f'{self.out_dir}/csa_{i_cycle}.csv', 'wt') as mj:
            for i, (i_mol, i_energy, i_gentype, i_qed_s, i_sa_s) in enumerate(zip(self.bank_pool, self.energy_bank_pool, self.bank_gen_type, self.bank_qed_s, self.bank_sa_s)):
                if first:
                    mj.write(',SMILES,GD3_Energy,Gentype,QED,SA\n')
                    mj.write(f'{i},{i_mol},{float(i_energy) * x},{i_gentype},{i_qed_s},{i_sa_s}\n')
                    first = False
                    continue
                mj.write(f'{i},{i_mol},{float(i_energy) * x},{i_gentype},{i_qed_s},{i_sa_s}\n')

        if i_cycle == self.max_opt_cycle or self.seed_cycle == self.n_seed_cycle:
            with open(f'{self.out_dir}/CSearch_result.smi','wt') as si:
                for i, i_mol  in enumerate(self.bank_pool):
                    si.write(f'{i_mol}\n')




        
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pdbid", type=str, required=True,
        help='Write the target pdbid')
    parser.add_argument(
        "-i", "--initial-bank", required=True, dest="smiles_fn",
        help='Initial bank smiles file')
    parser.add_argument(
        "-z", "--banksize", type=int, default=60,
        help='Initial bank size(Max:193)')
    parser.add_argument(
        "-b", "--building-blocks", required=True, dest="build_fn",
        help='Building blocks smiles file')
    parser.add_argument(
        "-s", "--seed_num", type=int, default=6,
        help='Number of Starting Seed')
    parser.add_argument(
        "-c", "--seed_cycle", type=int, default=2,
        help='Number of Seed Cycle')
    parser.add_argument(
        "-f", "--frtrack", type=str2bool, default=False,
        help='Tracking the origin of generated molecule')
    parser.add_argument(
        "-t", "--filter", type=str2bool, default=True,
        help='Filtering generated molecule by lipinski rule of 5')
    parser.add_argument(
        "-o", "--sc", type=str2bool, default=False,
        help='Use Synthetic Difficulty index of SCscore')
    parser.add_argument(
        "-n", "--num", type=int, default=0,
        help='Index for parallel executing')
    parser.add_argument(
        "-a", "--ann", type=int, default=20,
        help='Annealing step number')
    parser.add_argument(
        "-m", "--maxiter", type=int, default=50,
        help='Maxiter number')
    parser.add_argument(
        "-d", "--dmin", type=int, default=5,
        help='dmin denominator')
    args = parser.parse_args()
    smiles_fn = os.path.abspath(args.smiles_fn)
    build_fn = os.path.abspath(args.build_fn)

    start = time()
    job = Galaxy.initialize(title='Test_OptMol', mkdir=False)
    csa = CSA()
    csa.initialize_csa(job, smiles_fn, build_fn)
    csa.run()
    #csa.badseeds()
    end = time()
    print('time elapsed:', end - start)
