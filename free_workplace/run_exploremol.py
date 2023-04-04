import os,sys
import datetime
dt_now = datetime.datetime.now()
d_today = datetime.date.today()
pdblist = ['6M0K','5P9H']
now = str(dt_now.strftime('%m%d%H'))
for i in pdblist:
    with open(f'explore_{i}_genmol.sh','w') as g:
        g.write(
f'''#!/bin/sh
#SBATCH -p gpu.q
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --nodelist=nova013
#SBATCH --gpus=1
#SBATCH -J gen{i}
#SBATCH -o {i}_result/csa_{i}_{now}.log
python libfragcsa.py -p {i} -f True -t False''')
    os.system(f'sbatch explore_{i}_genmol.sh')

    
