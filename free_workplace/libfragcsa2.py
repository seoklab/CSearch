import os
import sys
import glob
import random
from typing import List

import Galaxy
import numpy as np
from scipy.spatial import distance as D
from opps.fragment_merge import gen_fr_mutation
from rdkit import Chem
from rdkit.Chem import RDConfig
from opps.libfilter import prepare_catalog_filters
from opps.libfilter import check_lipinski_filter
from opps.fragment_merge import gen_fr_mutation
from opps.fragment_merge import gen_crossover
from opps.fragment_merge import *
from opps.in_silico_reaction import (
    Reaction, get_dict_from_json_file, get_compl_mol_dict)
from opps.similarity_search import read_database_mol
from opps.energy_calculation_tab import energy_calc,qed_calc,sa_calc
from opps.visualization_tsne import make_fp_array,make_tsne_xy

N_PROC_DOCK = 1

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class CSA(object):
    def __init__(self, filter_lipinski=False, use_ML=True, ref_lig=None):
        #self.n_bank=192
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        #self.n_seed = 48
        self.n_bank = 50
        self.n_seed = 25
        #'dcut1': 2.0, # initial Dcut is the initial average diff / dcut1
        #'dcut2': 5.0, # minimum Dcut is the initial average diff / dcut2
        self.seed_mask = []  # any index in the list is not selected as a seed
        self.n_csa_iter = 1 
        self.n_seed_cycle = 10
        self.max_opt_cycle = 150
        self.dist_mat = np.zeros((self.n_bank, self.n_bank))
        self.use_ML = use_ML
        self.seed_cycle = 1
        self.g_array = []
        self.pdbid = "5P9H"
        self.n_opt_to_D_min = 100

        self.catalog_filters = prepare_catalog_filters(PAINS=True)
        self.filter_lipinski=filter_lipinski
        #if filter_PAINS.HasMatch(mol):
        #    continue
        ### TEMP, load model
        #with torch.no_grad():
        #    # ML model load
        #    model_s = [MyModel().to(device) for _ in MODEL_FN_S]
        #    [model.load_state_dict(torch.load(fn, map_location=device)['model_state_dict']) for model,fn in zip(model_s,MODEL_FN_S)]
        #    [model.eval() for model in model_s]
        #self.model_s = model_s

        self.ref_lig = ref_lig

    def initialize_csa(self, job, init_bank_smiles_fn, building_blocks_smiles_fn, n_proc=None):
        #self.prot_fn = prot_fn
        #self.cntr_R = cntr_R
        self.job = job
        n_proc = Galaxy.core.define_n_proc(n_proc)
        n_proc= 32
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
        self.building_block_setup(building_blocks_smiles_fn)
        self.setup_initial_Dcut()

    def run(self):
        self.write_bank(0)
        for i_csa_iter in range(self.n_csa_iter):
            for i_opt_cycle in range(1,self.max_opt_cycle+1):
                self.select_seeds()
                new_smiles_s, new_energy_s, new_mol_gen_type, new_qed_s, new_sa_s, operat_count = self.make_new_confs(i_opt_cycle)
                print(len(new_smiles_s))
                (a,b) = operat_count
                print('fragment_merge operator : %d, in silico rxn operator : %d'%(a,b))
                #print(type(new_smiles_s))
                #print(len(new_energy_s))
                #print(type(new_energy_s))
                #print(new_smiles_s)
                #print(new_energy_s)
                gn = make_fp_array(new_smiles_s, 'MACCS')
                self.g_array = np.concatenate((self.g_array, gn), axis=0)
                self.update_bank(new_smiles_s, new_energy_s, new_mol_gen_type, new_qed_s, new_sa_s)
                self.update_distance()
                self.update_functional_groups()
                self.print_log()
                print('CSA Run Number / Iteration / Seed cycle = %d / %d / %d'%(i_csa_iter, i_opt_cycle, \
                        self.seed_cycle))
                print('D_avg/D_cut/D_min=%.3f/%.3f/%.3f'%(self.D_avg, self.D_cut, self.D_cut_min))
                self.write_bank(i_opt_cycle)
                #update_distance
                self.D_cut = self.D_cut * self.xctdif
                self.D_cut = max(self.D_cut, self.D_min)
                print(f"D_cut : {self.D_cut}")
                if len(self.seed_mask) ==0:
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
                #smiles = smiles.replace('[CH]', 'C')

                mol = Molecule.from_smiles(smiles, source="INITIAL")
                if mol.RDKmol is None:
                    print(f'error processing {i_mol}th molecule')
                    continue
                mol.decompose()
                self.bank_pool.append(mol)
                if len(self.bank_pool) == self.n_bank:
                    break
        self.update_functional_groups()
        initial_smiles = [mol.smiles.replace("~", "") for mol in self.bank_pool]
        self.g_array = make_fp_array(initial_smiles,'MACCS')
        self.energy_bank_pool = energy_calc(initial_smiles, "csa", self.pdbid)
        self.bank_gen_type = ['o'] * self.n_bank
        self.bank_qed_s = qed_calc(initial_smiles)
        self.bank_sa_s = sa_calc(initial_smiles)
        print("initial bank energy:")
        print(self.energy_bank_pool)
        self.job.mkdir('cycle_0')
        smiles_block_s = []
        with open(smiles_fn, 'r') as fp:
            for line in fp:
                smiles_block_s.append(line)
        with open(f'/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/free_workplace/{self.pdbid}_result/csa_result.csv', 'w') as hj:
            hj.write(',max,min,average,in_silico,fragment\n')

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
        self.radical_mother = ''
        self.radical_father = ''
        self.radical_sons = []
        #store initial bank
        self.init_bank = self.bank_pool

    def building_block_setup(self,smiles_fn):
        self.building_pool: List[Molecule] = []
        
        with open(smiles_fn, 'r') as f_n:
            for i_mol, line in enumerate(f_n):
                smile_str = line.split()
                smiles_str = smile_str[0]
                smiles = smiles_str.replace('[NH3]', '[NH3+]')
                smiles = smiles.replace('[NH2]', '[NH2+]')
                smiles = smiles.replace('[NH]', '[NH+]')
                #smiles = smiles.replace('[CH]', 'C')

                mol = Molecule.from_smiles(smiles, source="INITIAL")
                if mol.RDKmol is None:
                    print(f'error processing {i_mol}th molecule')
                    continue
                mol.decompose()
                self.building_pool.append(mol)
                if len(self.building_pool) == 20*self.n_bank:
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

        self.D_cut = self.D_avg/2.0  # factor_init_D_cut
        self.D_cut_min = self.D_avg/5.0 # factor_min_D_cut

        self.D_min = self.D_avg/5.0

        nst = float(self.n_opt_to_D_min)/len(self.bank_pool)
        self.xctdif = (self.D_cut/self.D_min)**(-1.0/nst)

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

    def make_new_confs(self, i_cycle):
        seed_selected = self.select_seeds()
        #print(seed_selected)
        #print(str(seed_selected))
        in_silico_rxn_count = 0
        new_mol_gen_type = []
        new_mol_s: List[Molecule] = []
        mutation_all_s: List[Molecule] = []
        mutation_count = 0
        # generate molecules
        max_select_brics = 5
        # crossover, BRICS

        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            #seed_mol = self.init_bank[0]
            partner_mol = random.choice(self.init_bank)
            partner_mol2 = partner_mol
            (gen_RDKmol_s, Rad_mol_s, Hav_Rad) = gen_crossover(seed_mol, partner_mol,
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
                    r_s = ''.join(radical_son)
                    self.radical_sons += radical_son
                self.radical_mother = ''.join(radical_mother)
                self.radical_father = ''.join(radical_father)
                self.badseeds()
                continue
            if len(gen_RDKmol_s) > max_select_brics:
                gen_RDKmol_s = random.sample(gen_RDKmol_s, max_select_brics)

            for gen_RDKmol in gen_RDKmol_s:
                try:
                    new_mol = Molecule.from_rdkit(
                        gen_RDKmol, build_3d=True, source='BRICS')
                    new_mol_s.append(new_mol)
                    if len(new_mol_s) >= self.n_bank:
                        break
                except Exception:
                    continue

            if len(new_mol_s) >= self.n_bank:
                break
        frag_merge_count = len(new_mol_s)
        for numofrgmer in range(0,frag_merge_count):
            new_mol_gen_type.append('fr')
        #mutation,BRICS
        for i_seed in seed_selected:
            seed_mol = self.bank_pool[i_seed]
            #seed_mol = self.init_bank[0]
            #building_block_mol = random.choice(self.building_pool)
            #building_block_mol2 = random.choice(self.building_pool)
            (gen_RDKmol_s, Rad_mol_s, Hav_Rad) = gen_fr_mutation(seed_mol, self.building_pool,
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
                    r_s = ''.join(radical_son)
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
                        gen_RDKmol, build_3d=True, source='BRICS')
                    mutation_all_s.append(mutation_all)
                    if len(mutation_all_s) >= self.n_bank:
                        break
                except Exception:
                    continue

            if len(mutation_all_s) >= self.n_bank:
                break

            new_mol_s += mutation_all_s
            mutation_count += len(mutation_all_s)
            for insilico in range(0,mutation_count):
                new_mol_gen_type.append('mut')
        ## similarity searchi
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
        new_qed_s = qed_calc(
            [mol.smiles for mol in new_mol_s])
        new_sa_s = sa_calc(
            [mol.smiles for mol in new_mol_s])
        new_energy_s = energy_calc(
            [mol.smiles for mol in new_mol_s], "csa", self.pdbid)
        #new_energy_s = energy_calc(new_mol_s, multi=True)
        #print(new_energy_s[0])
        operat_count = (frag_merge_count,mutation_count)
        return new_mol_s, new_energy_s, new_mol_gen_type, new_qed_s, new_sa_s, operat_count

    def update_bank(self, new_mol_s: List[Molecule], new_energy_s: List[float], new_mol_gen_type: List[str], new_qed_s: List[float], new_sa_s: List[float]):
        for i, (i_mol, i_energy, i_gentype, i_qed_s, i_sa_s) in enumerate(zip(new_mol_s, new_energy_s, new_mol_gen_type, new_qed_s, new_sa_s)):
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
            if (min_dist < self.D_cut):
                if i_energy < self.energy_bank_pool[min_idx]:
                    #if i_energy + 0.000001 > self.energy_bank_pool[min_idx]:
                    #    continue
                    print('%d %.3f was replaced to %d %.3f in same group'
                          % (min_idx, self.energy_bank_pool[min_idx], i, i_energy))

                    print(f"before {self.bank_pool[min_idx]} after {i_mol}")
                    self.bank_pool[min_idx] = i_mol
                    self.energy_bank_pool[min_idx] = i_energy
                    self.bank_gen_type[min_idx] = i_gentype
                    self.bank_qed_s[min_idx] = i_qed_s
                    self.bank_sa_s[min_idx] = i_sa_s
                    #new_energy_s[min_idyx] = i_energy
                    if not min_idx in self.seed_mask:
                        self.seed_mask.append(min_idx)
            else:
                print(
                    '%d %.3f was replaced to %d %.3f in new group'
                    %(i_Emax_bank_u, self.energy_bank_pool[i_Emax_bank_u], i, i_energy))

                print(f"before {self.bank_pool[i_Emax_bank_u]} after {i_mol}")
                self.bank_pool[i_Emax_bank_u] = i_mol
                self.energy_bank_pool[i_Emax_bank_u] = i_energy
                self.bank_gen_type[i_Emax_bank_u] = i_gentype
                self.bank_qed_s[i_Emax_bank_u] = i_qed_s
                self.bank_sa_s[i_Emax_bank_u] = i_sa_s
                #new_energy_s[i_Emax_bank_u] = i_energy
                #new_energy_s[i_Emax_bank_u] = i_energy
                if not i_Emax_bank_u in self.seed_mask:
                    self.seed_mask.append(i_Emax_bank_u)

    def print_log(self):
        for i in range(self.n_bank):
            print('Bank %d:success'%(i+1))

    def write_bank(self, i_cycle):
        with open(f'{self.pdbid}_result/csa_%d.log'%i_cycle, 'wt') as fp:
            for i, (i_mol, i_energy, i_gentype, i_qed_s, i_sa_s) in enumerate(zip(self.bank_pool, self.energy_bank_pool, self.bank_gen_type, self.bank_qed_s, self.bank_sa_s)):
                #print(i_mol)
                #print(mol)
                #mol_energy = energy_calc(mol.smiles, 'single')
                fp.write(f'{i+1} GAenergy:{float(i_energy) * 100} Gentype:{i_gentype} QED:{i_qed_s} SA:{i_sa_s}\n')
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
            hj.write(f'{i_cycle},{MA},{MI},{AV},{QEDM},{QEDAV},{QEDmin},{SAM},{SAV},{SAm},{IS},{FR}\n')
    def badseeds(self):
        with open(f'{self.pdbid}_result/radicalparents.log', 'a') as gi:
            gi.write(f'{self.radical_mother} and {self.radical_father}\n')

if __name__=='__main__':
    import sys
    #prot_fn = Galaxy.core.FilePath(sys.argv[1])
#       smiles_fn = Galaxy.core.FilePath(sys.argv[2])
    smiles_fn = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/initial_bank_5P9H_0302.smi'
    smiles_fn = os.path.abspath(smiles_fn)
    build_fn = '/home/hakjean/galaxy2/developments/MolGen/MolGenCSA.git/data/Enamine_Fragment_Collection.smi'
    build_fn = os.path.abspath(build_fn)
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
    csa.initialize_csa(job, smiles_fn, build_fn)
    csa.run()
    csa.badseeds()
