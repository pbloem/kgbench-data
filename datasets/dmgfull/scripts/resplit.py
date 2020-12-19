import glob, sys, gzip

import rdflib as rdf
import numpy as np

"""
Quick script to produce new dataset splits, without loading the whole graph into RDF lib.
NB: These splits are not statified. 
"""

OLDSPLITDIR = '../raw/old_splits'
OUTDIR = '../raw'

VAL, TEST, META_TEST = 10_000, 20_000, 10_000
# training set is whatever is left over.

g = rdf.Graph()
for file in glob.glob(OLDSPLITDIR + '/*.nt.gz'):
    print(file)
    with gzip.open(file, 'rb') as f:
        g.parse(data=f.read(), format='nt')

total = len(g)

train = total - VAL - TEST - META_TEST

assert train > 0

print(f'train {train}, val {VAL}, test {TEST}, meta {META_TEST}')

np.random.seed(0)
bin = np.concatenate( [
    np.full((train,), 0),
    np.full((VAL,), 1),
    np.full((TEST,), 2),
    np.full((META_TEST,), 3) ], axis=0)

np.random.shuffle(bin) # in place

traing, testg, validg, mtestg = rdf.Graph(), rdf.Graph(), rdf.Graph(), rdf.Graph()

for i, triple in enumerate(g):
    if bin[i] == 0:
        traing.add(triple)
    elif bin[i] == 1:
        validg.add(triple)
    elif bin[i] == 2:
        testg.add(triple)
    elif bin[i] == 3:
        mtestg.add(triple)
    else:
        assert False

for filename, graph in zip(('train_set.nt.gz',
                            'test_set.nt.gz',
                            'valid_set.nt.gz',
                            'meta_set.nt.gz'),
                           (traing, testg, validg, mtestg)):

    with gzip.open(OUTDIR + '/' + filename, 'wb') as f:
        f.write(graph.serialize(format='nt'))
