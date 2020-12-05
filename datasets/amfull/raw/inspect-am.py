import hdt, gzip, tqdm

from collections import Counter

doc = hdt.HDTDocument('am-combined.hdt')
#
# triples, c = doc.search_triples('', '', '')
#
# relations = Counter()
#
# for s, p, o in tqdm.tqdm(triples, total=c):
#     relations[p] += 1
#
# for rel, count in relations.most_common():
#     print(f'{count: 5}: {rel}')

triples, c = doc.search_triples('', '', 'http://purl.org/collections/nl/am/t-10525')

for s, p, o in triples:
    print(s, p, o)

triples, c = doc.search_triples('http://purl.org/collections/nl/am/t-10525', '', '')

for s, p, o in triples:
    print(s, p, o)