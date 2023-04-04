#!/usr/bin/env python
import sys

sys.path.append('/home/hakjean/openbabel/openbabel-2.3.1/scripts/python')
from openbabel import openbabel
from openbabel import pybel
#from pybel import readfile

def draw(smi_fn, output_prefix=None):
    if output_prefix == None:
        output_prefix = smi_fn.split('.smi')[0]

    i_mol = 0
    for mol_pybel in pybel.readfile("smi", smi_fn):
        i_mol += 1
        mol_pybel.draw(show=False,filename = '%s_%d.png'%(output_prefix, i_mol))
        #mol_pybel.draw(show=False,filename = '%s_%d.png'%output_prefix)

if __name__=='__main__':
    mol2_fn = sys.argv[1]
    draw(mol2_fn)
    
#    import glob
#    mol2_fn_s = glob.glob('*/tmp.mol2')
#    for i, mol2_fn in enumerate(mol2_fn_s):
#        draw(mol2_fn, output_prefix='init_%d'%i)
