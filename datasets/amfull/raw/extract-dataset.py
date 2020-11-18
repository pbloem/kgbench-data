import hdt
import gzip
import pandas as pd
import numpy as np

import kgbench as kg

from tqdm import tqdm

"""
Extracts target labels.

"""

## Map from dataset category to coarse-grained classes.
map = {}

map['http://purl.org/collections/nl/am/t-14592'] = 'Books and documents'
# boekencollectie 	3479
# Book collection

map['http://purl.org/collections/nl/am/t-15459'] = 'Decorative arts'
# meubelcollectie 	3206
# Furniture

map['http://purl.org/collections/nl/am/t-15573'] = 'Decorative arts'
# glascollectie 	1028
# Glass

map['http://purl.org/collections/nl/am/t-15579'] = 'Decorative arts'
# textielcollectie 	7366
# Textiles

map['http://purl.org/collections/nl/am/t-15606'] = 'Decorative arts'
# keramiekcollectie 	5152

map['http://purl.org/collections/nl/am/t-16469'] = 'Metallic art'
# onedele metalen collectie 	797
# Non-noble metals

map['http://purl.org/collections/nl/am/t-22503'] = 'Visual art'
# prentencollectie 	22048
# Prints

map['http://purl.org/collections/nl/am/t-22504'] = 'Visual art'
# fotocollectie 	1563
# Photographs

map['http://purl.org/collections/nl/am/t-22505'] = 'Visual art'
# tekeningencollectie 	5455
# Drawings

map['http://purl.org/collections/nl/am/t-22506'] = 'Visual art'
# schilderijencollectie 	2672
# Paintings

map['http://purl.org/collections/nl/am/t-22507'] = 'Visual art'
# beeldencollectie 	943
# General image collection

map['http://purl.org/collections/nl/am/t-22508'] = 'Metallic art'
# edele metalencollectie 	3533
# Noble metals

map['http://purl.org/collections/nl/am/t-22509'] = 'Historical artifacts'
# penningen- en muntencollectie 	6440
# Coins etc.

map['http://purl.org/collections/nl/am/t-23765'] = 'Books and documents'
# documentencollectie 	533
# Document collection

map['http://purl.org/collections/nl/am/t-28650'] = 'Historical artifacts'
# archeologiecollectie 	582
# Archeaological artifacts

map['http://purl.org/collections/nl/am/t-31940'] = 'Metallic art'
# -- Onedele collectie 	3
# A small cetegory containing only room numbers from a defunct men's club

map['http://purl.org/collections/nl/am/t-32052'] = 'Historical artifacts'
# -- maten en gewichtencollectie 	536
# Measures and weight

map['http://purl.org/collections/nl/am/t-5504'] = 'Decorative arts'
# -- kunstnijverheidcollectie 	8087
# Arts and crafts

complete = hdt.HDTDocument('am-combined.hdt')

# the class relation
rel = 'http://purl.org/collections/nl/am/objectCategory'

data = []
triples, c = complete.search_triples('', rel, '')

for i, (s, _, o) in enumerate(triples):
    data.append([s, o])

df = pd.DataFrame(data, columns=['instance', 'label_original'])

df['cls_label'] = df.label_original.map(map)

df.cls_label = pd.Categorical(df.cls_label)
df['cls'] = df.cls_label.cat.codes

df.to_csv('all.csv', sep=',', index=False, header=True)
print('created dataframe.')

# * Split train, validation and test sets

# fixed seed for deterministic output
np.random.seed(0)

test_size = 10_000
val_size = 10_000
train_size =  len(df) - test_size - val_size

bin = np.concatenate( [np.full((train_size,), 0), np.full((val_size,), 1), np.full((val_size,), 2) ], axis=0)
np.random.shuffle(bin)

train = df[bin == 0]
train.to_csv('training.csv', sep=',', index=False, header=True)

val = df[bin == 1]
val.to_csv('validation.csv', sep=',', index=False, header=True)

test = df[bin == 1]
test.to_csv('testing.csv', sep=',', index=False, header=True)

print('created train, val, test split.')

stripped = hdt.HDTDocument('am-stripped.hdt')
triples, c = stripped.search_triples('', '', '')

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

triples, c = stripped.search_triples('', '', '')
print('Writing integer triples.')
with gzip.open('triples.int.csv.gz', 'wt') as file:

    for s, p, o in tqdm(triples, total=c):
        assert p != 'http://purl.org/collections/nl/am/objectCategory'
        assert p != 'http://purl.org/collections/nl/am/material'

        file.write(f'{e2i[s]}, {r2i[p]}, {e2i[o]}\n')








