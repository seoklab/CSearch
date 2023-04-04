import os,sys
with open('energy_sorted2.csv', 'w') as f:
    with open('energy_sorted.csv', 'r') as g:
        for i in g:
            f.write(','+i)

