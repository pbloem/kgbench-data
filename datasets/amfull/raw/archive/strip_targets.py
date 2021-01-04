import hdt

import gzip

import kgbench as kg

"""
Load the complete AM dataset and remove the target relation, together with the material relation (including which 
makes the task too easy). 
"""

document = hdt.HDTDocument('am-combined.hdt')

triples, c = document.search_triples('', '', '', limit=100)

print(c)
for s, p, o in triples:
    print(s, p, o)

# rel = rdf.term.URIRef("http://purl.org/collections/nl/am/objectCategory")
# g.remove((None, rel, None))
#
# rel = rdf.term.URIRef("http://purl.org/collections/nl/am/material")
# g.remove((None, rel, None))
#
# with gzip.open('am-stripped.nt.gz', 'wb') as output:
#     g.serialize(output, format='nt')
#
# g.close()