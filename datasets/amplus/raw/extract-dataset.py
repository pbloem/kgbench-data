import hdt
import gzip, sys, csv
import pandas as pd
import numpy as np

import kgbench as kg

from tqdm import tqdm

"""
Extracts target labels.

"""

def entity(ent):
    """
    Returns the value of an entity separated from its datatype ('represented by a string')

    :param ent:
    :return:
    """
    if ent.startswith('_'):
        return ent, 'blank_node'

    if ent.startswith('"'):
        if '^^' in ent:
            datatype, ent = ent[::-1].split('^^', maxsplit=1) # split once from the end
            datatype, ent = datatype[::-1], ent[::-1]
            datatype, ent = datatype[1:-1], ent[1:-1]
            return ent, datatype
        else:
            return ent[1:-1], 'none'

    else:
        assert ent.startswith('http') or ent.startswith('file') or ent.startswith('urn') or ent.startswith('mailto')
        # -- NB this assert only holds for this specific dataset.
        return ent, 'uri'

## Map from dataset category to coarse-grained classes.
map = {}

map['http://purl.org/collections/nl/am/t-14592'] = 'Books and Documents'
# boekencollectie 	3479
# Book collection

map['http://purl.org/collections/nl/am/t-15459'] = 'Decorative art'
# meubelcollectie 	3206
# Furniture

map['http://purl.org/collections/nl/am/t-15573'] = 'Decorative art'
# glascollectie 	1028
# Glass

map['http://purl.org/collections/nl/am/t-15579'] = 'Decorative art'
# textielcollectie 	7366
# Textiles

map['http://purl.org/collections/nl/am/t-15606'] = 'Decorative art'
# keramiekcollectie 	5152

map['http://purl.org/collections/nl/am/t-16469'] = 'Metallic art'
# onedele metalen collectie 	797
# Non-noble metals

map['http://purl.org/collections/nl/am/t-22503'] = 'Prints'
# prentencollectie 	22048
# Prints

map['http://purl.org/collections/nl/am/t-22504'] = 'Photographs'
# fotocollectie 	1563
# Photographs

map['http://purl.org/collections/nl/am/t-22505'] = 'Drawings'
# tekeningencollectie 	5455
# Drawings

map['http://purl.org/collections/nl/am/t-22506'] = 'Paintings'
# schilderijencollectie 	2672
# Paintings

map['http://purl.org/collections/nl/am/t-22507'] = 'Decorative art'
# beeldencollectie 	943
# Sculpture (?)

map['http://purl.org/collections/nl/am/t-22508'] = 'Metallic art'
# edele metalencollectie 	3533
# Noble metals

map['http://purl.org/collections/nl/am/t-22509'] = 'Historical artifacts'
# penningen- en muntencollectie 	6440
# Coins etc.

map['http://purl.org/collections/nl/am/t-28650'] = 'Historical artifacts'
# archeologiecollectie 	582
# Archeaological artifacts

map['http://purl.org/collections/nl/am/t-23765'] = 'Books and Documents'
# documentencollectie 	533
# Document collection

map['http://purl.org/collections/nl/am/t-31940'] = 'Metallic art'
# -- Onedele collectie 	3
# A small category containing only room numbers from a defunct men's club

map['http://purl.org/collections/nl/am/t-32052'] = 'Historical artifacts'
# -- maten en gewichtencollectie 	536
# Measures and weight

map['http://purl.org/collections/nl/am/t-5504'] = 'Decorative art'
# -- kunstnijverheidcollectie 	8087
# Arts and crafts

complete = hdt.HDTDocument('../../amfull/raw/am-combined.hdt')
# -- We use the AM combined data, because the target relations have already been stripped from amplus-all.xxx

# The class relation
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
print('Created dataframe. Class frequencies:')
print(df.cls_label.value_counts(normalize=False))
print(df.cls_label.value_counts(normalize=True))

# * Split train, validation and test sets

# fixed seed for deterministic output
np.random.seed(0)

meta_size = 20_000
test_size = 20_000
val_size = 20_000
train_size =  len(df) - test_size - val_size - meta_size

assert train_size > 0

print(f'train {train_size}, val {val_size}, test {test_size}, meta {meta_size}')

bin = np.concatenate( [
    np.full((train_size,), 0),
    np.full((val_size,), 1),
    np.full((test_size,), 2),
    np.full((meta_size,), 3) ], axis=0)

np.random.shuffle(bin) # in place

train = df[bin == 0]
train.to_csv('training.csv', sep=',', index=False, header=True)

val = df[bin == 1]
val.to_csv('validation.csv', sep=',', index=False, header=True)

test = df[bin == 2]
test.to_csv('testing.csv', sep=',', index=False, header=True)

test = df[bin == 3]
test.to_csv('meta-testing.csv', sep=',', index=False, header=True)

print('created train, val, test, meta split.')
print('Creating dictionaries.')

stripped = hdt.HDTDocument('amplus-stripped.hdt')

entities = set()
relations = set()
datatypes = set()

triples, c = stripped.search_triples('', '', '')
for s, p, o in tqdm(triples, total=c):
    datatypes.add(entity(s)[1])
    datatypes.add(entity(o)[1])

i2d = list(datatypes)
i2d.sort()
d2i = {d:i for i, d in enumerate(i2d)}

triples, c = stripped.search_triples('', '', '')
for s, p, o in tqdm(triples, total=c):

    se, sd = entity(s)
    oe, od = entity(o)

    entities.add((se, sd))
    entities.add((oe, od))

    relations.add(p)

i2e = list(entities)
i2r = list(relations)

i2e.sort(); i2r.sort()

df = pd.DataFrame(enumerate(i2d), columns=['index', 'datatype'])
df.to_csv('datatypes.int.csv', index=False, header=True)

ent_data = [(i, dt, ent) for i, (ent, dt) in enumerate(i2e)]
df = pd.DataFrame(ent_data, columns=['index', 'datatype', 'label'])
df.to_csv('entities.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

df = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
df.to_csv('relations.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

e2i = {e: i for i, e in enumerate(i2e)}
r2i = {r: i for i, r in enumerate(i2r)}

for file in ['training', 'testing', 'validation', 'meta-testing']:
    df = pd.read_csv(file + '.csv')
    classes = df.cls
    instances = df.instance
    intinstances = instances.map(lambda ent : e2i[(ent, 'uri')])
    # -- all instances have datatype uri

    pd.concat([intinstances, classes], axis=1).to_csv(file + '.int.csv', index=False, header=True)

## Convert stripped triples to integer triples
triples, c = stripped.search_triples('', '', '')
print('Writing integer triples.')
with gzip.open('triples.int.csv.gz', 'wt') as file:

    for s, p, o in tqdm(triples, total=c):
        assert p != 'http://purl.org/collections/nl/am/objectCategory'
        assert p != 'http://purl.org/collections/nl/am/material'

        sp = entity(s)
        op = entity(o)

        file.write(f'{e2i[sp]}, {r2i[p]}, {e2i[op]}\n')








