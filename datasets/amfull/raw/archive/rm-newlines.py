import rdflib as rdf
import gzip

import kgbench as kg

"""
Load the complete AM dataset with a proper parser and store it as a sequence of n-triples guarenteed to 
be easy to parse (one triple per line)
"""

with gzip.open('am-combined.nt.gz', 'rb') as input:
    g = kg.load_rdf(input, name='am-combined', format='nt')

with gzip.open('am-combined-nnl.gz', 'wt') as output:

    for s, p, o in g.triples((None, None, None)):
        print(f'{} {} {}')



g.close()