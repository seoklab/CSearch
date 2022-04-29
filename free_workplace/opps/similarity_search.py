import sys

import numpy as np
from openbabel import pybel

from .fragment_merge import Molecule
from .libfilter import check_catalog_filters, check_lipinski_filter

fn_database_mol = '/home/neclasic/Screen/DB_Compound/Commercial/Asinex/All/2020-01-BioDesign.sdf'
#fn_database_mol = '/home/neclasic/Screen/DB_Compound/ZINC/druglike_90/druglike_90.mol2'


def read_database_mol(fn_database_mol):
    mol_s=list(pybel.readfile("sdf", fn_database_mol))
    mol_fp_s = []
    first = True
    for mol in mol_s:
        mol = pybel.readstring("smi",str(mol))
        mol_fp_s.append(mol.calcfp(fptype="fp4"))
    return mol_s, mol_fp_s


def pick_similar_compound(seed_mol, database_mol_s, database_fp_s, filters=None, filter_lipinski=False):
    print(seed_mol)
    #seed_fp = seed_mol.calcfp()
    seed_imol = pybel.readstring("smi",str(seed_mol))
    #seed_fp = FingerprintMols.FingerprintMol(seed_imol)
    seed_fp = seed_imol.calcfp(fptype="fp4")
    tani_s = []
    for mol_fp in database_fp_s:
        tani_coef = seed_fp|mol_fp
        tani_s.append(tani_coef)
    tani_s = np.array(tani_s)

    # pick similiar 500 compounds
    similar_indices = np.argpartition(-tani_s, 500)
    np.random.shuffle(similar_indices)
    # pick random 5 compounds
    new_mol_s = []
    for idx in similar_indices:
        #try:
        smiles = database_mol_s[idx].write()
        new_mol = Molecule.from_smiles(smiles, build_3d=True)
        if not filters == None:
            check_catalog = check_catalog_filters(new_mol.RDKmol, filters)
            if check_catalog:
                #print('filtered_PAINS')
                continue
        if filter_lipinski:
            check_lipinski = check_lipinski_filter(new_mol.RDKmol)
            if check_lipinski:
                #print('filtered_LIP')
                continue
        new_mol_s.append(new_mol)
       # except:
       #     continue
        if len(new_mol_s) == 2:
            break
    return new_mol_s

def main():
    import time
    query_mol2_fn = sys.argv[1]
    query_mol = next(pybel.readfile("mol2", query_mol2_fn))
    query_mol = pybel.readstring("smi",str(query_mol))
    query_fp = query_mol.calcfp(fptype="fp4")
    mol_s=list(pybel.readfile("sdf", fn_database_mol))

    tani_s = []
    t1 = time.time()
    for mol in mol_s:
        mol = pybel.readstring("smi",str(mol))
        fp_mol = mol.calcfp(fptype="fp4")
        tani_coef = query_fp|fp_mol
        tani_s.append(tani_coef)
    t2 = time.time()
    print(t2-t1)

    tani_s = np.array(tani_s)
    t1 = time.time()
    similarity_indices = np.argsort(-tani_s)
    print(tani_s[similarity_indices[0]])
    print(tani_s[similarity_indices[1]])
    t2 = time.time()

    new_mol_s = []

    i_mol = 0
    for idx in similarity_indices:
        smiles = mol_s[idx].write()
        new_mol = Molecule.from_smiles(smiles, build_3d=True)
        #if not filters == None:
        #    check_catalog = check_catalog_filters(new_mol.RDKmol, filters)
        #    if check_catalog:
        #        #print('filtered_PAINS')
        #        continue
        #if filter_lipinski:
        check_lipinski = check_lipinski_filter(new_mol.RDKmol)
        if check_lipinski:
            #print('filtered_LIP')
            continue
        i_mol+=1
        new_mol_s.append(new_mol)
       # except:
       #     continue
        with open('mol_%d.mol2'%i_mol, 'wt') as fp:
            fp.writelines(new_mol.mol2_block)
        if len(new_mol_s) == 192:
            break

#    t1 = time.time()
#    similarity_indices = np.argpartition(tani_s, 1001)
#    #similarity_indices = np.argpartition(-tani_s, 1001)
#    print(tani_s[similarity_indices[-1]])
#    print(tani_s[similarity_indices[-2]])
#    t2 = time.time()

    return

if __name__=='__main__':
    main()
