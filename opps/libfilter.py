from rdkit.Chem import FilterCatalog
from rdkit.Chem import Descriptors, Lipinski, Crippen

def prepare_catalog_filters(PAINS=False, NIH=False, ZINC=False, ALL=False):
    filters = []
    if ALL:
        params_ALL = FilterCatalog.FilterCatalogParams()
        params_ALL.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.ALL)
        filter_ALL = FilterCatalog.FilterCatalog(params_ALL)
        filters.append(filter_ALL)
        return filters
    if PAINS:
        params_PAINS = FilterCatalog.FilterCatalogParams()
        params_PAINS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        filter_PAINS = FilterCatalog.FilterCatalog(params_PAINS)
        filters.append(filter_PAINS)
    if NIH:
        params_NIH = FilterCatalog.FilterCatalogParams()
        params_NIH.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
        filter_NIH = FilterCatalog.FilterCatalog(params_NIH)
        filters.append(filter_NIH)
    if ZINC:
        params_ZINC = FilterCatalog.FilterCatalogParams()
        params_ZINC.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.ZINC)
        filter_ZINC = FilterCatalog.FilterCatalog(params_ZINC)
        filters.append(filter_ZINC)
    return filters

def check_lipinski_filter(mol):
    '''
    This runs a Strict Lipinski filter. Lipinski filter refines for orally
    available drugs. It filters molecules by Molecular weight (MW), the
    number of hydrogen donors, the number hydrogen acceptors, and the logP
    value.

    This is a strict Lipinski which means a ligand must pass all the
    requirements.

    To pass the Lipinski filter a molecule must be:
        MW: Max 500 dalton
        Number of H acceptors: Max 10
        Number of H donors: Max 5
        logP Max +5.0

    If you use the Lipinski Filter please cite: C.A. Lipinski et al.
    Experimental and computational approaches to estimate solubility and
    permeability in drug discovery and development settings Advanced Drug
    Delivery Reviews, 46 (2001), pp. 3-26
    '''

    exact_mwt = Descriptors.ExactMolWt(mol)
    if exact_mwt > 500:
        return True

    num_hydrogen_bond_donors = Lipinski.NumHDonors(mol)
    if num_hydrogen_bond_donors > 5:
        return True

    num_hydrogen_bond_acceptors = Lipinski.NumHAcceptors(mol)
    if num_hydrogen_bond_acceptors > 10:
        return True

    mol_log_p = Crippen.MolLogP(mol)
    if mol_log_p > 5:
        return True

    # Passed all filters
    return False


def check_catalog_filters(RDKmol, filters):
    return any(filter.HasMatch(RDKmol) for filter in filters)


# def main():
#     import sys
#     mol2_fn = sys.argv[1]
#     from fragment_merge import Molecule_Pool
#     mol_pool=Molecule_Pool(mol2_fn)
#     filters = prepare_catalog_filters(PAINS=True, ALL=True)
#     for mol in mol_pool:
#         check_catalog = check_catalog_filters(mol.RDKmol, filters)
#         check_lipinski = check_lipinski_filter(mol.RDKmol)
#         print(check_catalog, check_lipinski)

# if __name__=='__main__':
#     main()
