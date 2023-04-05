import os
import sys
import random
import argparse
from time import time
from typing import List

import Galaxy
import numpy as np
from scipy.spatial import distance as D
from rdkit import Chem
from rdkit.Chem import RDConfig

from opps.libs.utils import str2bool
from opps.libfilter import prepare_catalog_filters, check_lipinski_filter
from opps.fragment_merge import (
    Molecule, gen_fr_mutation, gen_crossover, make_fragments_set,
    calc_tanimoto_distance)
from opps.energy_calculation_tab import energy_calc, qed_calc, sa_calc

N_PROC_DOCK = 1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CSA(object):
    def __init__(self, use_ML=True, ref_lig=None):
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        self.n_bank = 50
        self.n_seed = args.seed_num
        self.seed_mask = []  # any index in the list is not selected as a seed
        self.n_csa_iter = 1
        self.n_seed_cycle = args.seed_cycle
        self.max_opt_cycle = 50
        self.dist_mat = np.zeros((self.n_bank, self.n_bank))
        self.use_ML = use_ML
        self.seed_cycle = 1
        self.pdbid = args.pdbid
        self.n_opt_to_D_min = 50
        self.catalog_filters = prepare_catalog_filters(PAINS=True)
        self.filter_lipinski = args.filter
        self.ref_lig = ref_lig

    def initialize_csa(self, job, init_bank_smiles_fn,
                       building_blocks_smiles_fn, n_proc=None):
        self.job = job
        n_proc = Galaxy.core.define_n_proc(n_proc)
        n_proc = 32
        self.n_proc = n_proc

        # initial bank
        self.read_initial_bank(init_bank_smiles_fn)
        self.building_block_setup(building_blocks_smiles_fn)
        self.setup_initial_Dcut()

    def run(self):
        self.write_bank(0)
        for i_csa_iter in range(self.n_csa_iter):
            for i_opt_cycle in range(1, self.max_opt_cycle + 1):
                self.select_seeds()
                (new_smiles_s, new_energy_s, new_mol_gen_type, new_qed_s,
                 new_sa_s, operat_count) = self.make_new_confs(i_opt_cycle)
                print(len(new_smiles_s))
                (a, b) = operat_count
                print('Mutation operator : %d, Crossover operator : %d'%(b,a))
                self.update_bank(new_smiles_s, new_energy_s, new_mol_gen_type,
                                 new_qed_s, new_sa_s)
                self.update_distance()
                self.update_functional_groups()
                self.print_log()
                print('CSA Run Number / Iteration / Seed cycle = %d / %d / %d'
                      % (i_csa_iter, i_opt_cycle, self.seed_cycle))
                print('D_avg/D_cut/D_min=%.3f/%.3f/%.3f'
                      % (self.D_avg, self.D_cut, self.D_cut_min))
                self.write_bank(i_opt_cycle)
                # update_distance
                self.D_cut = self.D_cut * self.xctdif
                self.D_cut = max(self.D_cut, self.D_min)
                print(f"D_cut : {self.D_cut}")
                if len(self.seed_mask) == 0:
                    if self.seed_cycle == self.n_seed_cycle:
                        return
                    self.seed_cycle += 1
                    self.select_seeds()

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
        initial_mols = [mol.RDKmol for mol in self.bank_pool]
        self.energy_bank_pool = energy_calc(initial_mols, "csa", self.pdbid)
        self.bank_gen_type = ['o'] * self.n_bank
        self.bank_qed_s = qed_calc(initial_mols)
        self.bank_sa_s = sa_calc(initial_mols)
        if args.frtrack:
            self.bank_frg = [make_fragments_set(i) for i in self.bank_pool]
        smiles_block_s = []
        with open(smiles_fn, 'r') as fp:
            for line in fp:
                smiles_block_s.append(line)

        out_dir = f"{self.pdbid}_result"
        self.job.mkdir(out_dir, cd=False)
        with open(f'{out_dir}/csa_result.csv', 'w') as f:
            f.write(',max,min,average,in_silico,fragment\n')

        self.radical_mother = ''
        self.radical_father = ''
        self.radical_sons = []
        # store initial bank
        self.init_bank = self.bank_pool

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
                mol.decompose()
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

        self.D_cut = self.D_avg / 2.0  # factor_init_D_cut
        self.D_cut_min = self.D_avg / 5.0  # factor_min_D_cut

        self.D_min = self.D_avg / 5.0

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

        # pick from unused
        seed_selected_unused = random.sample(
            self.seed_mask, n_seed_from_unused)
        # store used seed
        for i_seed in seed_selected_unused:
            self.seed_mask.remove(i_seed)

        # pick rest from used
        used_mask = [
            i for i in range(1, self.n_seed + 1) if i not in self.seed_mask]
        seed_selected_used = random.sample(used_mask, n_seed_from_used)

        seed_selected = seed_selected_unused + seed_selected_used
        return seed_selected

    def update_functional_groups(self):
        for mol in self.bank_pool:
            mol.determine_functional_groups()

    def make_new_confs(self, i_cycle):
        seed_selected = self.select_seeds()
        new_mol_gen_type = []
        new_mol_s: List[Molecule] = []
        mutation_all_s: List[Molecule] = []
        mutation_count = 0
        # generate molecules
        max_select_brics = 5
        # crossover, BRICS

        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            partner_mol = random.choice(self.init_bank)
            partner_mol2 = partner_mol
            gen_RDKmol_s, Rad_mol_s, Hav_Rad = gen_crossover(
                seed_mol, partner_mol,
                filters=self.catalog_filters,
                filter_lipinski=self.filter_lipinski)
            if Hav_Rad:
                radical_mother = []
                radical_father = []
                radical_son = []
                radical_mother += str(self.bank_pool[i_seed])
                try:
                    radical_father += str(partner_mol2)
                except:
                    radical_father += partner_mol2
                for rad_mol in Rad_mol_s:
                    radical_son.append(str(rad_mol))
                    self.radical_sons += radical_son
                self.radical_mother = ''.join(radical_mother)
                self.radical_father = ''.join(radical_father)
                self.badseeds()
                continue
            if len(gen_RDKmol_s) > max_select_brics:
                gen_RDKmol_s = random.sample(gen_RDKmol_s, max_select_brics)

            for gen_RDKmol in gen_RDKmol_s:
                try:
                    new_mol = Molecule.from_rdkit(gen_RDKmol, source='BRICS')
                    new_mol_s.append(new_mol)
                    if len(new_mol_s) >= self.n_bank:
                        break
                except Exception:
                    continue

            if len(new_mol_s) >= self.n_bank:
                break
        frag_merge_count = len(new_mol_s)
        for _ in range(0, frag_merge_count):
            new_mol_gen_type.append('fr')
        # mutation,BRICS
        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            gen_RDKmol_s, Rad_mol_s, Hav_Rad = gen_fr_mutation(
                seed_mol, self.building_pool,
                filters=self.catalog_filters,
                filter_lipinski=self.filter_lipinski)
            if Hav_Rad:
                radical_mother = []
                radical_father = []
                radical_son = []
                radical_mother += str(self.bank_pool[i_seed])
                try:
                    radical_father += str(partner_mol2)
                except:
                    radical_father += partner_mol2
                for rad_mol in Rad_mol_s:
                    radical_son.append(str(rad_mol))
                    self.radical_sons += radical_son
                self.radical_mother = ''.join(radical_mother)
                self.radical_father = ''.join(radical_father)
                self.badseeds()
                continue
            if len(gen_RDKmol_s) > max_select_brics:
                gen_RDKmol_s = random.sample(gen_RDKmol_s, max_select_brics)

            for gen_RDKmol in gen_RDKmol_s:
                try:
                    mutation_all = Molecule.from_rdkit(
                        gen_RDKmol, source='BRICS')
                    mutation_all_s.append(mutation_all)
                    if len(mutation_all_s) >= self.n_bank:
                        break
                except Exception:
                    continue

            if len(mutation_all_s) >= self.n_bank:
                break

            new_mol_s += mutation_all_s
            mutation_count += len(mutation_all_s)
            for _ in range(0, mutation_count):
                new_mol_gen_type.append('mut')

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

            if i_energy >= self.energy_bank_pool[i_Emax_bank_u]:
                continue

            # check lipinski filter
            if check_lipinski_filter(i_mol.RDKmol):
                continue
            # check closest bank member

            dist_s = [calc_tanimoto_distance(bank_mol, i_mol)
                      for bank_mol in self.bank_pool]
            min_idx = np.argmin(dist_s)
            min_dist = dist_s[min_idx]
            i_mol.RDKmol.UpdatePropertyCache()
            Chem.GetSymmSSSR(i_mol.RDKmol)
            i_mol.decompose()  # move to somewhere

            # replace current bank
            if (min_dist < self.D_cut):
                if i_energy < self.energy_bank_pool[min_idx]:
                    print('B%d %.3f was replaced to %d %.3f in same group'
                          % (min_idx, self.energy_bank_pool[min_idx],
                             i, i_energy))

                    print(f"before {self.bank_pool[min_idx]} after {i_mol}")
                    self.bank_pool[min_idx] = i_mol
                    self.energy_bank_pool[min_idx] = i_energy
                    self.bank_gen_type[min_idx] = i_gentype
                    self.bank_qed_s[min_idx] = i_qed_s
                    self.bank_sa_s[min_idx] = i_sa_s
                    if args.frtrack:
                        self.bank_frg[min_idx] = make_fragments_set(
                            self.bank_pool[min_idx])
                        print(
                            f'New molecule Fragments:{self.bank_frg[min_idx]}')
                    if not min_idx in self.seed_mask:
                        self.seed_mask.append(min_idx)
            else:
                print('B%d %.3f was replaced to %d %.3f in new group' %
                      (i_Emax_bank_u, self.energy_bank_pool[i_Emax_bank_u],
                       i, i_energy))

                print(f"before {self.bank_pool[i_Emax_bank_u]} after {i_mol}")
                self.bank_pool[i_Emax_bank_u] = i_mol
                self.energy_bank_pool[i_Emax_bank_u] = i_energy
                self.bank_gen_type[i_Emax_bank_u] = i_gentype
                self.bank_qed_s[i_Emax_bank_u] = i_qed_s
                self.bank_sa_s[i_Emax_bank_u] = i_sa_s
                if args.frtrack:
                    self.bank_frg[i_Emax_bank_u] = make_fragments_set(
                        self.bank_pool[i_Emax_bank_u])
                    print('New molecule Fragments:' +
                          str(self.bank_frg[i_Emax_bank_u]))
                if not i_Emax_bank_u in self.seed_mask:
                    self.seed_mask.append(i_Emax_bank_u)

    def print_log(self):
        for i in range(self.n_bank):
            print('Bank %d:success'%(i+1))
        if args.frtrack:
            hajime = True
            for sets in self.bank_frg:
                if hajime:
                    emptyset = sets
                    hajime = False
                    continue
                emptyset = emptyset.union(sets)
            print(f'Number of Fragments:{len(emptyset)}')

    def write_bank(self, i_cycle):
        with open(f'{self.pdbid}_result/csa_%d.log' % i_cycle, 'wt') as fp:
            for i, (i_mol, i_energy, i_gentype, i_qed_s, i_sa_s) in enumerate(
                    zip(self.bank_pool, self.energy_bank_pool,
                        self.bank_gen_type, self.bank_qed_s, self.bank_sa_s)):
                fp.write(f'{i+1} GAenergy:{float(i_energy) * 100} '
                         f'Gentype:{i_gentype} QED:{i_qed_s} SA:{i_sa_s}\n')
                fp.write(f'{i_mol}\n')
        MI = max(self.energy_bank_pool) * 100
        MA = min(self.energy_bank_pool) * 100
        AV = 100 * sum(self.energy_bank_pool) / self.n_bank
        QEDM = max(self.bank_qed_s)
        QEDAV = sum(self.bank_qed_s) / self.n_bank
        QEDmin = min(self.bank_qed_s)
        SAM = max(self.bank_sa_s)
        SAV = sum(self.bank_sa_s) / self.n_bank
        SAm = min(self.bank_sa_s)
        IS = self.bank_gen_type.count('i_s')
        FR = self.bank_gen_type.count('fr')
        with open(f'{self.pdbid}_result/csa_result.csv', 'a') as hj:
            hj.write(f'{i_cycle},{MA},{MI},{AV},{QEDM},{QEDAV},'
                     f'{QEDmin},{SAM},{SAV},{SAm},{IS},{FR}\n')

    def badseeds(self):
        with open(f'{self.pdbid}_result/radicalparents.log', 'a') as gi:
            gi.write(f'{self.radical_mother} and {self.radical_father}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pdbid", type=str, required=True,
        help='Write the target pdbid')
    parser.add_argument(
        "-i", "--initial-bank", required=True, dest="smiles_fn",
        help='Initial bank smiles file')
    parser.add_argument(
        "-b", "--building-blocks", required=True, dest="build_fn",
        help='Building blocks smiles file')
    parser.add_argument(
        "-s", "--seed_num", type=int, default=25,
        help='Number of Starting Seed')
    parser.add_argument(
        "-c", "--seed_cycle", type=int, default=5,
        help='Number of Seed Cycle')
    parser.add_argument(
        "-f", "--frtrack", type=str2bool, default=False,
        help='Tracking the origin of generated molecule')
    parser.add_argument(
        "-t", "--filter", type=str2bool, default=True,
        help='Filtering generated molecule by lipinski rule of 5')
    args = parser.parse_args()
    smiles_fn = os.path.abspath(args.smiles_fn)
    build_fn = os.path.abspath(args.build_fn)

    start = time()
    job = Galaxy.initialize(title='Test_OptMol', mkdir=False)
    csa = CSA()
    csa.initialize_csa(job, smiles_fn, build_fn)
    csa.run()
    csa.badseeds()
    end = time()
    print('time elapsed:', end - start)
