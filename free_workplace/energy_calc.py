import os,sys
from rdkit import Chem
from opps.energy_calculation_tab import energy_calc
with open('/home/hakjean/galaxy2/developments/DB_Clustering/DrugBank/DrugBank_approved.smi', 'r') as r:
    for lin in r:
        lin = lin.split()
        #smi_list.append(lin[0])
        if len(lin[0]) < 10:
            continue
        if Chem.MolFromSmiles(lin[0]) == None:
            continue
        try:
            energy = energy_calc(lin[0], 'single')
        except:
            continue
        with open('energy.csv', 'a') as g:
            g.write(f'{lin[0]},{energy}\n')
