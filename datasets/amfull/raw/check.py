import hdt, gzip
import rdflib as rdf

from kgbench import tic, toc

doc = hdt.HDTDocument('am-combined.hdt')

triples, c = doc.search_triples('', '', '')

e = set()
lits = blanks = uris = other = 0

for s, p, o in triples:
    
    if p == "http://purl.org/collections/nl/am/objectCategory":
        pass
    elif p == "http://purl.org/collections/nl/am/material":
        pass
    else:
        e.add(s)
        e.add(o)

lits = blanks = uris = 0
others = set()
for x in e:
    if x.startswith('\"'):
        lits += 1
    elif x.startswith('_'):
        blanks += 1
    elif x.startswith('http') or x.startswith('file'):
        uris += 1
    else:
        others.add(x)

print('total entities', len(e))
print('   literals', lits)
print('     blanks', blanks)
print('       uris', uris)
print('     others', len(others))
print('       sum', lits + blanks + uris + len(others))

for o in others:
    print(others)

with gzip.open('/Users/peter/Dropbox/datasets/RDF/am/am-combined.nt.gz', 'r') as file:

    g = rdf.Graph()
    tic()
    g.parse(file, format='nt')
    print(f'loaded {toc()}s.')

    rel = rdf.term.URIRef("http://purl.org/collections/nl/am/objectCategory")
    g.remove((None, rel, None))

    rel = rdf.term.URIRef("http://purl.org/collections/nl/am/material")
    g.remove((None, rel, None))

    e = set()
    for s, p, o in g.triples((None, None, None)):
        # s, o = str(s), str(o)
        e.add(s)
        e.add(o)

    lits = blanks = uris = 0
    others = set()
    for x in e:
        if type(x) == rdf.Literal:
            lits += 1
        elif type(x) == rdf.BNode:
            blanks += 1
        elif type(x) == rdf.URIRef:
            uris += 1
        else:
            others.add(x)

    print('total entities', len(e))
    print('   literals', lits)
    print('     blanks', blanks)
    print('       uris', uris)
    print('     others', len(others))
    print('       sum', lits + blanks + uris + len(others))

    for o in others:
        print(others)

    g.close()