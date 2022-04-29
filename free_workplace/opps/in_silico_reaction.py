import glob
import random

from rdkit import Chem
from rdkit.Chem import AllChem

from .libfilter import check_catalog_filters, check_lipinski_filter
from .fragment_merge import Molecule, Molecule_Pool

global i_reaction
i_reaction = 0

class Reaction(object):
    def __init__(self, reaction_dict):
        self.reaction_name = reaction_dict["reaction_name"]
        self.functional_groups = reaction_dict["functional_groups"]
        self.group_smarts = reaction_dict["group_smarts"]
        self.num_reactants = reaction_dict["num_reactants"]
        self.reaction_string = reaction_dict["reaction_string"]
        self.RDKrxn = AllChem.ReactionFromSmarts(self.reaction_string)
        self.RDKrxn.Initialize()

    def __repr__(self):
        return self.reaction_name

    def run_reaction(self, reactants, filters=None,
                     filter_lipinski=False, build_3d=False):
        global i_reaction
        i_reaction += 1
        RDK_reactants = []

        i = 0
        for functional_group in self.functional_groups:
            react = reactants[functional_group]
            RDK_reactants.append(react.RDKmol)

            i+=1
            #react.draw(output_fn='test_reaction/reagent_%s_%d.png'%(self.reaction_name,i))

        RDK_reactants = tuple(RDK_reactants)
        products = self.RDKrxn.RunReactants(RDK_reactants)
        products = [product[0] for product in products]
        new_mol_s = []
        for RDKmol in products:
            #print(Chem.MolToSmiles(RDKmol), 'product')
            #try:
            smiles = Chem.MolToSmiles(RDKmol)
            #print(smiles)
            try:
                product = Molecule.from_smiles(
                    smiles, build_3d=build_3d, source="REACTION")
                if filters is not None:
                    check_catalog = check_catalog_filters(
                        product.RDKmol, filters)
                    if check_catalog:
                        continue
                if filter_lipinski:
                    check_lipinski = check_lipinski_filter(product.RDKmol)
                    if check_lipinski:
                        continue
            #product = Molecule(RDKmol=RDKmol)
                new_mol_s.append(product)
            except BaseException:
                continue
        i = 0
        #for product in products:
        #i+=1
        # product.draw(output_fn='test_reaction/product_%s_%d.png'%(self.reaction_name,i))

        return new_mol_s

    def check_reaction_components(self, mol):
        groups_missing = []
        groups_found = []

        for functional_group in self.functional_groups:
            if mol.HasFunctionalGroup[functional_group]:
                groups_found.append(functional_group)
            else:
                groups_missing.append(functional_group)
        return groups_found, groups_missing


def get_dict_from_json_file(fn_json):
    #import json
    #with open(fn_json,'r') as fp:
    #    json_dict = json.loads(fp.read())
    import yaml
    with open(fn_json,'r') as fp:
        json_dict = yaml.safe_load(fp.read())
    return json_dict

def get_compl_mol_dict(fn_compl_mol_s):
    compl_mol_dict = {}
    for fn_compl_mol in fn_compl_mol_s:
        group_name = fn_compl_mol.split('/')[-1].split('.')[0]
        with open(fn_compl_mol, 'r') as fp:
            smi_s = [line.split()[0] for line in fp.readlines()]
        compl_mol_s = [Molecule.from_smiles(smiles) for smiles in smi_s]
        compl_mol_dict[group_name] = compl_mol_s
    return compl_mol_dict

if __name__=='__main__':
    fn_func_json = '/home/neclasic/Screen/GalaxyMolOpt/data/reaction_libraries/all_rxns/All_Rxns_functional_groups.json'
    functional_group_dict = get_dict_from_json_file(fn_func_json)

    fn_rxn_lib = '/home/neclasic/Screen/GalaxyMolOpt/data/reaction_libraries/all_rxns/All_Rxns_rxn_library.json'
    reaction_dict = get_dict_from_json_file(fn_rxn_lib)

    fn_compl_mol_s = glob.glob('/home/neclasic/Screen/GalaxyMolOpt/data/reaction_libraries/all_rxns/complementary_mol_dir/*.smi')
    compl_mol_dict = get_compl_mol_dict(fn_compl_mol_s)

    mol_pool = Molecule_Pool('test.mol2')
    mol_pool.determine_functional_groups()
    test_mol = mol_pool[0]

    reaction_s = []
    for reaction_name in reaction_dict:
        reaction_s.append(Reaction(reaction_dict[reaction_name]))

    for reaction in reaction_s:
        groups_found, groups_missing = reaction.check_reaction_components(test_mol)
        print(reaction.functional_groups)
        print(len(groups_found), len(groups_missing))
        if len(groups_found) == 0:
            continue
        group_picked = random.choice(groups_found)
        reactants = {}
        reactants[group_picked] = test_mol

        for group in reaction.functional_groups:
            if group_picked == group:
                continue
            compl_mol = random.choice(compl_mol_dict[group])
            reactants[group] = compl_mol

        products = reaction.run_reaction(reactants)
        print (products)
