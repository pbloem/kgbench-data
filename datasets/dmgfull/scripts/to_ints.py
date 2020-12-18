import hdt, gzip, csv
from tqdm import tqdm
import pandas as pd
from rdflib import Graph

import kgbench as kg


"""

"""

doc = hdt.HDTDocument('DMGfull_stripped.hdt')

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
df.to_csv('nodetypes.int.csv', index=False, header=True)

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

# Load test/train/valid
g_train = Graph()
with gzip.open('./train_set.nt.gz', 'rb') as gzf:
    g_train.parse(gzf, format='nt')
g_test = Graph()
with gzip.open('./test_set.nt.gz', 'rb') as gzf:
    g_test.parse(gzf, format='nt')
g_valid = Graph()
with gzip.open('./valid_set.nt.gz', 'rb') as gzf:
    g_valid.parse(gzf, format='nt')

c2i = dict()
i = 0
for g in [g_train, g_test, g_valid]:
    classes = set(g.objects())
    for c in classes:
        if c not in c2i.keys():
            c2i[c] = i
            i += 1

with open('all.int.csv', 'w') as allfile:
    allwriter = csv.writer(allfile, delimiter=',')
    allwriter.writerow(['index', 'class'])

    with open('training.int.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['index', 'class'])
        for s, p, o in g_train.triples((None, None, None)):
            s_idx = e2i[s]
            o_idx = c2i[o]

            writer.writerow([s_idx, o_idx])
            allwriter.writerow([s_idx, o_idx])

    with open('testing.int.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['index', 'class'])
        for s, p, o in g_test.triples((None, None, None)):
            s_idx = e2i[s]
            o_idx = c2i[o]

            writer.writerow([s_idx, o_idx])
            allwriter.writerow([s_idx, o_idx])

    with open('validation.int.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['index', 'class'])
        for s, p, o in g_valid.triples((None, None, None)):
            s_idx = e2i[s]
            o_idx = c2i[o]

            writer.writerow([s_idx, o_idx])
            allwriter.writerow([s_idx, o_idx])

