"""
Remove the target triples from the AM data
"""

import hdt, gzip
from tqdm import tqdm
import rdflib as rdf

def f(x : str):
    if x.startswith('_'): # blank node, leave as is
        return x
    if x.startswith('"'): # literal, rm newlines and escape internal quotes and slashes
        x = x[1:-1]
        x = x.replace('\n', '.newline').replace('\r', '.cr')
        lit = rdf.Literal(x)
        return lit.n3()

    else: # url, enclose with <>
        assert x.startswith('http') or x.startswith('file'), x
        return f'<{x}>'

doc = hdt.HDTDocument('am-combined.hdt')


triples, c = doc.search_triples('','','')

with gzip.open('am-stripped.nt.gz', 'wt') as file:

    stripped_oc = 0
    stripped_material = 0

    entities = set()
    unformatted = set()

    for s, p, o in tqdm(triples, total=c):
        if p == 'http://purl.org/collections/nl/am/objectCategory':
            stripped_oc += 1
        elif p == 'http://purl.org/collections/nl/am/material':
            stripped_material += 1
        else:
            s, p, o = f(s), f(p), f(o)
            file.write(f'{s} {p} {o} . \n')
            entities.add(s); entities.add(o)
            unformatted.add(s); unformatted.add(o)

print(f'Wrote stripped data. Removed {stripped_oc} oc edges, and {stripped_material} material edges.')
print('     entities: ', len(entities))
print('  unformatted: ', len(unformatted))
