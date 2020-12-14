"""
Remove the target triples from the AM data
"""

import hdt, gzip
from tqdm import tqdm
import rdflib as rdf
import sys

def f(x : str):
    if x.startswith('_'): # blank node, leave as is
        return x
    if x.startswith('"'): # literal, rm newlines and escape internal quotes and slashes
        datatype=None
        if '"^^<' in x:
            datatype, x = x[::-1].split('^^', maxsplit=1)
            datatype, x = datatype[::-1], x[::-1]
            datatype = datatype[1:-1]
        x = x[1:-1]

        x = x.replace('\n', '.newline').replace('\r', '.cr')
        lit = rdf.Literal(x, datatype=datatype)
        return lit.n3()

    else: # url, enclose with <>
        assert x.startswith('http') or x.startswith('file'), x
        return f'<{x}>'

whitelist = {
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
    'http://www.europeana.eu/schemas/edm/object' # links to b64 encoded images
}


doc = hdt.HDTDocument('amplus-all.hdt')
triples, c = doc.search_triples('','','')

with gzip.open('amplus-stripped.nt.gz', 'wt') as file:

    stripped = 0

    entities = set()
    unformatted = set()

    for s, p, o in tqdm(triples, total=c):
        if \
           p not in whitelist or \
          (p == 'http://www.europeana.eu/schemas/edm/object' and o.startswith('http')): # filter out img URLs
            stripped += 1
        else:
            s, p, o = f(s), f(p), f(o)
            file.write(f'{s} {p} {o} . \n')

            entities.add(s); entities.add(o)
            unformatted.add(s); unformatted.add(o)

print(f'Wrote stripped data. Removed {stripped} edges.')
print('     entities: ', len(entities))
print('  unformatted: ', len(unformatted))
