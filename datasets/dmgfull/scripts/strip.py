
import hdt

import gzip, sys, tqdm

import kgbench as kg

doc = hdt.HDTDocument('../raw/DMGfull_unstripped.hdt')
triples, c = doc.search_triples('','','')

rm = ['http://purl.org/dc/terms/subject']

with gzip.open('../raw/DMGFull-stripped.nt.gz', 'wt') as file:

    stripped = 0

    for s, p, o in tqdm.tqdm(triples, total=c):
        if p not in rm:
            file.write(f'{kg.n3(s)} {kg.n3(p)} {kg.n3(o)} . \n')
        else:
            stripped += 1

print(f'Wrote stripped data. Removed {stripped} edges.')
