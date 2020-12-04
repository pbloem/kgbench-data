import hdt, gzip, sys
from tqdm import tqdm
import pandas as pd


"""

"""

doc = hdt.HDTDocument('aifb_stripped.hdt')
triples, c = doc.search_triples('', '', '')

i2e = set()
i2r = set()

for s, p, o in tqdm(triples, total=c):
    i2e.add(s)
    i2r.add(p)
    i2e.add(o)

i2e = list(i2e)
i2r = list(i2r)

i2e.sort(); i2r.sort()
# -- this is requires for the script to be deterministic

e2i = {e:i for i, e in enumerate(i2e)}
r2i = {r:i for i, r in enumerate(i2r)}

# print(i2e[5], e2i[i2e[5]])

# Write entity and relation maps to CSV
df = pd.DataFrame(enumerate(i2e), columns=['index', 'label'])
df.to_csv('entities.int.csv', index=False, header=True)

df = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
df.to_csv('relations.int.csv', index=False, header=True)

# Write triples to CSV
print('Writing integer triples.')
triples, c = doc.search_triples('', '', '')
with gzip.open('triples.int.csv.gz', 'wt') as file:

    for s, p, o in tqdm(triples, total=c):
        file.write(f'{e2i[s]}, {r2i[p]}, {e2i[o]}\n')

# Load test/train
trainin = pd.read_csv('trainingSet.tsv', sep='\t')
testin  = pd.read_csv('testSet.tsv', sep='\t')

all = pd.concat([trainin, testin])
print(all.columns)

all.label_affiliation = pd.Categorical(all.label_affiliation)
all['cls'] = all.label_affiliation.cat.codes
all['instance'] = all.person.map(e2i)

all = all[['instance', 'cls']]
all.to_csv('all.int.csv', index=False, header=True)

train, val, test = all[:104], all[104:140], all[140:177]

print(len(train), len(val), len(test))

train.to_csv('training.int.csv', index=False, header=False)
val.to_csv('validation.int.csv', index=False, header=False)
test.to_csv('testing.int.csv', index=False, header=False)


