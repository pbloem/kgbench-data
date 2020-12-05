import hdt, gzip, tqdm

from collections import Counter


def getlabel(x, doc):

    triples, c = doc.search_triples(x, 'http://www.w3.org/2004/02/skos/core#prefLabel', '')
    return next(triples)[2]

# ---
# triples, c = doc.search_triples('', '', '')
#
# relations = Counter()
#
# for s, p, o in tqdm.tqdm(triples, total=c):
#     relations[p] += 1
#
# for rel, count in relations.most_common():
#     print(f'{count: 5}: {rel}')

# ---
# triples, c = doc.search_triples('', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', '')
#
# types = Counter()
# for s, p, o in triples:
#     types[o] += 1
#
# for type, ct in types.most_common():
#     print(f'{ct: 5}: {type}')

doc = hdt.HDTDocument('am-combined.hdt')
triples, c = doc.search_triples('', 'http://www.w3.org/2004/02/skos/core#broader', '')

types = Counter()
for s, p, o in triples:
    slabel = getlabel(s, doc)
    olabel = getlabel(o, doc)

    types[slabel] += 1
    types[olabel] += 1

for type, ct in types.most_common():
    print(f'{ct: 5}: {type}')