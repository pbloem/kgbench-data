import hdt
import gzip, sys, csv
import pandas as pd
import numpy as np

import kgbench as kg

from tqdm import tqdm

"""
Extracts target labels.

"""

doc = hdt.HDTDocument('dblp-reg.hdt')

triples, c = doc.search_triples('', 'http://www.w3.org/2000/01/rdf-schema#seeAlso', '')

cits = pd.read_csv('citation_counts.csv', header=None)
cits.columns = ['doi', 'num_citations']

ax = cits.num_citations.hist()
fig = ax.get_figure()
fig.savefig('citation hist.png')

print(cits.num_citations.value_counts())
print(f'Median number of citations: {cits.num_citations.median()}')

instances = []

# collect instane URIs
for doi in cits.doi:
    triples, c = doc.search_triples('', 'http://www.w3.org/2000/01/rdf-schema#seeAlso', 'http://dx.doi.org/' + doi)

    inst = None
    for s, p, o in triples:
        inst = s
        break

    assert inst is not None, f'Instance {doi} not found among triples'
    instances.append(inst)

cits['instance'] = instances
cits['cls'] = (cits.num_citations > cits.num_citations.median()).astype(int) # class label

cits = cits[['doi','instance','num_citations','cls']]

cits.to_csv('all.csv', sep=',', index=False, header=True)
print('Created dataframe. Class frequencies:')
print(cits.cls.value_counts(normalize=False))
print(cits.cls.value_counts(normalize=True))

# * Split train, validation and test sets

# fixed seed for deterministic output
np.random.seed(0)

meta_size = 10_000
test_size = 20_000
val_size = 10_000
train_size =  len(cits) - test_size - val_size - meta_size

assert train_size > 0

print(f'train {train_size}, val {val_size}, test {test_size}, meta {meta_size}')

bin = np.concatenate( [
    np.full((train_size,), 0),
    np.full((val_size,), 1),
    np.full((test_size,), 2),
    np.full((meta_size,), 3) ], axis=0)

np.random.shuffle(bin) # in place

train = cits[bin == 0]
train.to_csv('training.csv', sep=',', index=False, header=True)

val = cits[bin == 1]
val.to_csv('validation.csv', sep=',', index=False, header=True)

test = cits[bin == 2]
test.to_csv('testing.csv', sep=',', index=False, header=True)

test = cits[bin == 3]
test.to_csv('meta-testing.csv', sep=',', index=False, header=True)

print('created train, val, test, meta split.')
print('Creating dictionaries.')

entities = set()
relations = set()
datatypes = set()

triples, c = doc.search_triples('', '', '')
for s, p, o in tqdm(triples, total=c):
    datatypes.add(kg.entity_hdt(s)[1])
    datatypes.add(kg.entity_hdt(o)[1])

i2d = list(datatypes)
i2d.sort()
d2i = {d:i for i, d in enumerate(i2d)}

triples, c = doc.search_triples('', '', '')
for s, p, o in tqdm(triples, total=c):

    se, sd = kg.entity_hdt(s)
    oe, od = kg.entity_hdt(o)

    entities.add((se, sd))
    entities.add((oe, od))

    relations.add(p)

i2e = list(entities)
i2r = list(relations)

i2e.sort(); i2r.sort()

df = pd.DataFrame(enumerate(i2d), columns=['index', 'annotation'])
df.to_csv('annotation-types.int.csv', index=False, header=True)

ent_data = [(i, dt, ent) for i, (ent, dt) in enumerate(i2e)]
df = pd.DataFrame(ent_data, columns=['index', 'annotation', 'label'])
df.to_csv('nodes.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

df = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
df.to_csv('relations.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

e2i = {e: i for i, e in enumerate(i2e)}
r2i = {r: i for i, r in enumerate(i2r)}

for file in ['training', 'testing', 'validation', 'meta-testing']:
    df = pd.read_csv(file + '.csv')
    classes = df.cls
    instances = df.instance
    intinstances = instances.map(lambda ent : e2i[(ent, 'iri')])
    # -- all instances have datatype iri

    pd.concat([intinstances, classes], axis=1).to_csv(file + '.int.csv', index=False, header=True)

## Convert stripped triples to integer triples
triples, c = doc.search_triples('', '', '')
print('Writing integer triples.')
with gzip.open('triples.int.csv.gz', 'wt') as file:

    for s, p, o in tqdm(triples, total=c):

        sp = kg.entity_hdt(s)
        op = kg.entity_hdt(o)

        file.write(f'{e2i[sp]}, {r2i[p]}, {e2i[op]}\n')








