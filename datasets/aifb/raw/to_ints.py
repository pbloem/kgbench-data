import hdt, gzip, sys, csv
from tqdm import tqdm
import pandas as pd

import kgbench as kg


"""

"""

doc = hdt.HDTDocument('aifb_stripped.hdt')

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
# -- this is required for the script to be deterministic

df = pd.DataFrame(enumerate(i2d), columns=['index', 'annotation'])
df.to_csv('annotation-types.int.csv', index=False, header=True)

ent_data = [(i, dt, ent) for i, (ent, dt) in enumerate(i2e)]
df = pd.DataFrame(ent_data, columns=['index', 'annotation', 'label'])
df.to_csv('nodes.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

df = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
df.to_csv('relations.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

e2i = {e:i for i, e in enumerate(i2e)}
r2i = {r:i for i, r in enumerate(i2r)}

# Write triples to CSV
print('Writing integer triples.')
triples, c = doc.search_triples('', '', '')
with gzip.open('triples.int.csv.gz', 'wt') as file:

    for s, p, o in tqdm(triples, total=c):

        sp = kg.entity_hdt(s)
        op = kg.entity_hdt(o)

        file.write(f'{e2i[sp]}, {r2i[p]}, {e2i[op]}\n')

# Load test/train
trainin = pd.read_csv('trainingSet.tsv', sep='\t')
testin  = pd.read_csv('testSet.tsv', sep='\t')

all = pd.concat([trainin, testin])
print(all.columns)

all.label_affiliation = pd.Categorical(all.label_affiliation)
all['cls'] = all.label_affiliation.cat.codes
all['instance'] = all.person.map(lambda ent : e2i[(ent, 'iri')] )

all = all[['instance', 'cls']]
all.to_csv('all.int.csv', index=False, header=True)

train, val, test = all[:104], all[104:140], all[140:177]

print(len(train), len(val), len(test))

train.to_csv('training.int.csv', index=False, header=False)
val.to_csv('validation.int.csv', index=False, header=False)
test.to_csv('testing.int.csv', index=False, header=False)


