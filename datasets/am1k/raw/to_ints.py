import rdflib as rdf
import gzip
import pandas as pd
import os

import kgbench as kg

from tqdm import tqdm

"""
Takes the source files for the original AM task (with 1K labeled nodes) and converts them to the canonical integer-based
format used here. The entity and relation dictionaries and the integer triples are taken from the amfull preprocessing.

The original train/test split was manually split further into train/val/test, by splitting the original training data 
again. Loading in final mode recovers the original split.
"""

i2e = pd.read_csv('../entities.int.csv').label.tolist()
i2r = pd.read_csv('../relations.int.csv').label.tolist()
# these can be copied from amfull

e2i = {e:i for i, e in enumerate(i2e)}
r2i = {r:i for i, r in enumerate(i2r)}

classes = pd.read_csv('training.tsv', sep='\t').label_category.tolist()
i2c = list(set(classes))
c2i = {c:i for i, c in enumerate(i2c)}

for file in ['training', 'testing', 'validation']:
    df = pd.read_csv(file + '.tsv', sep='\t')
    classes = df.label_category.map(c2i)
    instances = df.proxy
    intinstances = instances.map(e2i)

    pd.concat([intinstances, classes], axis=1).to_csv(file + '.int.csv', index=False, header=False)




