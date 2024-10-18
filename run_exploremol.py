import os,sys
import subprocess as sp
import datetime
dt_now = datetime.datetime.now()
d_today = datetime.date.today()
pdblist = ['3TI5','4MKC','5P9H','6M0K']
bslist = [60]
#filist = [True,False]
seednumratio = [0.1]
seedcyclelist = [2]
maxiter = [50]
#sclist = ['nostrcen','0302','noscfr']
#sclist = ['0302']
now = str(dt_now.strftime('%m%d%H'))
#si = 'drugbank'
banklist = ["drugspace"]
nstlist = [20]
nst = 20
dmin = [4,5,10]
dm = 5
#for si in sclist:
for bs in bslist:
    for sn in seednumratio:
        for si in banklist:
            for sc in seedcyclelist:
                for mx in maxiter:
                    #for si in banklist:
                    for pdbid in pdblist:
                        if mx == 100:
                            nst = nst*2
                        parameter = "pdb : " + pdbid + " initbank :" + si + " banknum :" + str(bs) + " seed_num : " + str(bs*sn) + " seed_cycle :" + str(sc) + " maxiter :" + str(mx) + " nst :" + str(nst) + str(dm)
                        param = pdbid + si + str(bs) + str(int(bs*sn)) +'_'+ str(sc) +'_'+ str(mx) + str(nst) + str(dm)
                        for num in range(1,101):
                            if num < 10:
                                num = '0' + str(num)
                            print(parameter+str(num))
                            with open(f'explore_{param}.sh','w') as g:
                                g.write(f'''#!/bin/sh
#SBATCH -p normal.q
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --nodelist=star0[01-46]
#SBATCH -J gen{pdbid}
#SBATCH -o Result/{pdbid}/{d_today}/csa_{param}_{num}.log
#SBATCH --nice=100000000
python CSearch.py -p {pdbid} -i data/Iinitial_bank_{pdbid}_{si}.smi -z {bs} -b data/Enamine_Fragment_Collection_single.smi -s {int(bs*sn)} -c {sc} -n {num} -a {nst} -m {mx} -d {dm}''')
                            sp.run(['sbatch', f'./explore_{param}.sh'])
