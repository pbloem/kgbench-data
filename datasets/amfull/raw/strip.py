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

blacklist = [
'http://purl.org/collections/nl/am/objectCategory',     # target relation
'http://purl.org/collections/nl/am/material',           # target relation

'http://purl.org/collections/nl/am/AHMTextsType',       # reveals target

'http://purl.org/collections/nl/am/partsTitle',         # frequent relations
'http://www.w3.org/2000/01/rdf-schema#subPropertyOf',
'http://rdfs.org/ns/void#dataDump',
'http://purl.org/dc/terms/title',
'http://www.swi-prolog.org/rdf/library/source',
'http://rdfs.org/ns/void#subset',
'http://www.w3.org/2000/01/rdf-schema#subClassOf',
'http://purl.org/collections/nl/am/documentationNotes',
'http://purl.org/collections/nl/am/lang',

'http://purl.org/collections/nl/am/alternativenumber',
'http://purl.org/collections/nl/am/alternativeNumberType',
'http://purl.org/collections/nl/am/alternativeNumber',

'http://purl.org/collections/nl/am/documentationTitle',
'http://purl.org/collections/nl/am/documentation',
'http://purl.org/collections/nl/am/documentationTitleLref',
'http://purl.org/collections/nl/am/documentationPageReference',
'http://purl.org/collections/nl/am/documentationSortyear',
'http://purl.org/collections/nl/am/documentationAuthor',

'http://purl.org/collections/nl/am/priref',

'http://purl.org/collections/nl/am/acquisitionMethod',
'http://purl.org/collections/nl/am/acquisitionDate',

]

blacklist = set(blacklist)

doc = hdt.HDTDocument('am-combined.hdt')
triples, c = doc.search_triples('','','')

with gzip.open('am-stripped.nt.gz', 'wt') as file:

    stripped = 0

    entities = set()
    unformatted = set()

    for s, p, o in tqdm(triples, total=c):
        if p in blacklist:
            stripped += 1
        else:
            s, p, o = f(s), f(p), f(o)
            file.write(f'{s} {p} {o} . \n')

            entities.add(s); entities.add(o)
            unformatted.add(s); unformatted.add(o)

print(f'Wrote stripped data. Removed {stripped} edges.')
print('     entities: ', len(entities))
print('  unformatted: ', len(unformatted))
