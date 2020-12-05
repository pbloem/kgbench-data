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

# blacklist = [
#     'http://purl.org/collections/nl/am/objectCategory',     # target relation
#     'http://purl.org/collections/nl/am/material',           # target relation
#
#     'http://purl.org/collections/nl/am/AHMTextsType',       # reveals target
#
#     'http://purl.org/collections/nl/am/partsTitle',         # frequent relations
#     'http://www.w3.org/2000/01/rdf-schema#subPropertyOf',
#     'http://rdfs.org/ns/void#dataDump',
#     'http://purl.org/dc/terms/title',
#     'http://www.swi-prolog.org/rdf/library/source',
#     'http://rdfs.org/ns/void#subset',
#     'http://www.w3.org/2000/01/rdf-schema#subClassOf',
#     'http://purl.org/collections/nl/am/documentationNotes',
#     'http://purl.org/collections/nl/am/lang',
#
#     'http://purl.org/collections/nl/am/alternativenumber',
#     'http://purl.org/collections/nl/am/alternativeNumberType',
#     'http://purl.org/collections/nl/am/alternativeNumber',
#
#     'http://purl.org/collections/nl/am/documentationTitle',
#     'http://purl.org/collections/nl/am/documentation',
#     'http://purl.org/collections/nl/am/documentationTitleLref',
#     'http://purl.org/collections/nl/am/documentationPageReference',
#     'http://purl.org/collections/nl/am/documentationSortyear',
#     'http://purl.org/collections/nl/am/documentationAuthor',
#
#     'http://purl.org/collections/nl/am/priref',
#
#     'http://purl.org/collections/nl/am/acquisitionMethod',
#     'http://purl.org/collections/nl/am/acquisitionDate',
# ]
#
# blacklist = set(blacklist)


whitelist = [
    'http://purl.org/collections/nl/am/title',
    'http://www.openarchives.org/ore/terms/proxyIn',
    'http://purl.org/collections/nl/am/maker',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#value',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2004/02/skos/core#broader',
    'http://purl.org/collections/nl/am/birthDateEnd',
    'http://purl.org/collections/nl/am/deathDateEnd',
    'http://purl.org/collections/nl/am/birthPlace',
    'http://purl.org/collections/nl/am/deathPlace',
    'http://purl.org/collections/nl/am/nationality',
    'http://purl.org/collections/nl/am/biography', # ~60% acc

    'http://www.openarchives.org/ore/terms/proxyFor',
    'http://purl.org/collections/nl/am/dimension',
    'http://purl.org/collections/nl/am/dimensionType',
    'http://purl.org/collections/nl/am/dimensionUnit',
    'http://purl.org/collections/nl/am/dimensionValue',
    'http://purl.org/collections/nl/am/termType', # ~75% acc

    'http://purl.org/collections/nl/am/currentLocationType',
    'http://purl.org/collections/nl/am/currentLocationDateEnd',
    'http://purl.org/collections/nl/am/productionDateStart',
    'http://purl.org/collections/nl/am/productionDateEnd',
    'http://purl.org/collections/nl/am/currentLocationNotes',
    'http://purl.org/collections/nl/am/exhibition',
    'http://purl.org/collections/nl/am/exhibitionLref',
    'http://purl.org/collections/nl/am/exhibitionTitle',
    'http://purl.org/collections/nl/am/reproductionFormat',
    'http://purl.org/collections/nl/am/exhibitionOrganiser',
    'http://purl.org/collections/nl/am/wasPresentAt',
    'http://purl.org/collections/nl/am/exhibitionVenue',
    'http://purl.org/collections/nl/am/exhibitionDateStart',
    'http://purl.org/collections/nl/am/exhibitionDateEnd',
]

whitelist= set(whitelist)

doc = hdt.HDTDocument('am-combined.hdt')
triples, c = doc.search_triples('','','')

with gzip.open('am-stripped.nt.gz', 'wt') as file:

    stripped = 0

    entities = set()
    unformatted = set()

    for s, p, o in tqdm(triples, total=c):
        if p not in whitelist:
            stripped += 1
        else:
            s, p, o = f(s), f(p), f(o)
            file.write(f'{s} {p} {o} . \n')

            entities.add(s); entities.add(o)
            unformatted.add(s); unformatted.add(o)

print(f'Wrote stripped data. Removed {stripped} edges.')
print('     entities: ', len(entities))
print('  unformatted: ', len(unformatted))
