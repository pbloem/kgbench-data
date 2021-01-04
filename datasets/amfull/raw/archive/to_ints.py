"""
Reads a graph in n-triples format, together with the TSVs describing target classes, and prodcues integer only versions
of the task, together with the int-to-label mappings.

"""

import hdt
import gzip
import pandas as pd
import os

import kgbench as kg

from tqdm import tqdm

document = hdt.HDTDocument('am-combined.hdt')
triples, c = document.search_triples('', '', '')

entities = set()
relations = set()

print('Creating dictionaries.')
for s, p, o in tqdm(triples, total=c):
    entities.add(str(s))
    entities.add(str(o))
    relations.add(str(p))

i2e = list(entities)
i2r = list(relations)

df = pd.DataFrame(enumerate(i2e), columns=['index', 'label'])
df.to_csv('entities.int.csv', index=False, header=True)

df = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
df.to_csv('relations.int.csv', index=False, header=True)

e2i = {e:i for i, e in enumerate(i2e)}
r2i = {r:i for i, r in enumerate(i2r)}

for file in ['training', 'testing', 'validation']:
    df = pd.read_csv(file + '.csv')
    classes = df.cls
    instances = df.instance
    intinstances = instances.map(e2i)

    pd.concat([intinstances, classes], axis=1).to_csv(file + '.int.csv', index=False, header=False)

## Convert stripped triples to integer triples

document = hdt.HDTDocument('am-combined.hdt')
triples, c = document.search_triples('', '', '')

print('Writing stripped data.')
with gzip.open('../triples.int.csv.gz', 'wt') as file:

    stripped_oc = 0
    stripped_material = 0
    for s, p, o in tqdm(triples, total=c):

        if p == 'http://purl.org/collections/nl/am/objectCategory':
            stripped_oc +=1
        elif p == 'http://purl.org/collections/nl/am/material':
            stripped_material +=1
        else:
            file.write(f'{s}, {p}, {o}\n')

print(f'Wrote stripped data. Removed {stripped_oc} oc edges, and {stripped_material} material edges.')



