import gzip, sys, csv, rdflib
import pandas as pd
import numpy as np
from rdflib import Graph

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
    if type(ent) == rdflib.term.Literal:
        datatype = str(ent.datatype)
        if not datatype:
            datatype = 'unknown'
    else:
        assert str(ent).startswith('http'), "{} does not start with allowed string"
        datatype = 'iri'
    return str(ent), datatype


print("Parsing graph")
complete = Graph()
with gzip.open('md_raw.nt.gz', 'rb') as f:
    complete.parse(f, format="nt")

# The class relation
gender = rdflib.term.URIRef("http://www.wikidata.org/prop/direct/P21")

print("Searching for rel")
data = []
for i, (s, r, o) in enumerate(complete):
    if r == gender:
        data.append([str(s), str(o)])

df = pd.DataFrame(data, columns=['instance', 'cls_label'])

df.cls_label = pd.Categorical(df.cls_label)
df['cls'] = df.cls_label.cat.codes

df.to_csv('../all.csv', sep=',', index=False, header=True)
print('Created dataframe. Class frequencies:')
print(df.cls_label.value_counts(normalize=False))
print(df.cls_label.value_counts(normalize=True))

# * Split train, validation and test sets

# fixed seed for deterministic output
np.random.seed(0)

meta_size = 1000
test_size = 10000
val_size = 1000
train_size = len(df) - test_size - val_size - meta_size

assert train_size > 0

print(f'train {train_size}, val {val_size}, test {test_size}, meta {meta_size}')

bin = np.concatenate( [
    np.full((train_size,), 0),
    np.full((val_size,), 1),
    np.full((test_size,), 2),
    np.full((meta_size,), 3) ], axis=0)

np.random.shuffle(bin) # in place

train = df[bin == 0]
train.to_csv('../training.csv', sep=',', index=False, header=True)

val = df[bin == 1]
val.to_csv('../validation.csv', sep=',', index=False, header=True)

test = df[bin == 2]
test.to_csv('../testing.csv', sep=',', index=False, header=True)

test = df[bin == 3]
test.to_csv('../meta-testing.csv', sep=',', index=False, header=True)

print('Created train, val, test, meta split.')

stripped = complete
stripped.remove((None, gender, None))

print('Creating dictionaries.')
entities = set()
relations = set()
datatypes = set()

for s, p, o in tqdm(stripped):
    datatypes.add(entity(s)[1])
    datatypes.add(entity(o)[1])

i2d = list(datatypes)
i2d.sort()
d2i = {d:i for i, d in enumerate(i2d)}

for s, p, o in tqdm(stripped):

    se, sd = entity(s)
    oe, od = entity(o)

    entities.add((se, sd))
    entities.add((oe, od))

    relations.add(p)

i2e = list(entities)
i2r = list(relations)

i2e.sort(); i2r.sort()

df = pd.DataFrame(enumerate(i2d), columns=['index', 'annotation'])
df.to_csv('../annotation-types.int.csv', index=False, header=True)

ent_data = [(i, dt, ent) for i, (ent, dt) in enumerate(i2e)]
df = pd.DataFrame(ent_data, columns=['index', 'annotation', 'label'])
df.to_csv('../nodes.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

df = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
df.to_csv('../relations.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

e2i = {e: i for i, e in enumerate(i2e)}
r2i = {r: i for i, r in enumerate(i2r)}

for file in ['training', 'testing', 'validation', 'meta-testing']:
    df = pd.read_csv("../" + file + '.csv')
    classes = df.cls
    instances = df.instance
    intinstances = instances.map(lambda ent : e2i[(ent, 'iri')])
    # -- all instances have datatype uri

    pd.concat([intinstances, classes], axis=1).to_csv("../" + file + '.int.csv', index=False, header=False)

## Convert stripped triples to integer triples
print('Writing integer triples.')
with gzip.open('../triples.int.csv.gz', 'wt') as file:

    for s, p, o in tqdm(stripped):
        assert str(p) != 'http://www.wikidata.org/prop/direct/P21'

        sp = entity(s)
        op = entity(o)

        file.write(f'{e2i[sp]}, {r2i[p]}, {e2i[op]}\n')
