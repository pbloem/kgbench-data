#!/usr/bin/env python

import csv
import gzip
import sys

# https://github.com/Callidon/pyHDT
import hdt
from tqdm import tqdm
import pandas as pd
from rdflib import Graph

import kgbench as kg



def generate_csv_context(doc):
    entities = set()
    relations = set()
    datatypes = set()

    triples, c = doc.search_triples('', '', '')
    for s, p, o in tqdm(triples, total=c):
        datatypes.add(kg.entity_hdt(s)[1])
        datatypes.add(kg.entity_hdt(o)[1])

    i2d = list(datatypes)
    i2d.sort()

    triples, c = doc.search_triples('', '', '')
    for s, p, o in tqdm(triples, total=c):

        se, sd = kg.entity_hdt(s)
        oe, od = kg.entity_hdt(o)

        entities.add((se, sd))
        entities.add((oe, od))

        relations.add(p)

    i2e = list(entities)
    i2r = list(relations)

    i2e.sort(); i2r.sort()
    # -- this is required for the script to be deterministic

    df = pd.DataFrame(enumerate(i2d), columns=['index', 'annotation'])
    df.to_csv('nodetypes.int.csv', index=False, header=True)

    ent_data = [(i, dt, ent) for i, (ent, dt) in enumerate(i2e)]
    df = pd.DataFrame(ent_data, columns=['index', 'annotation', 'label'])
    df.to_csv('nodes.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

    df = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
    df.to_csv('relations.int.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

    e2i = {e:i for i, e in enumerate(i2e)}
    r2i = {r:i for i, r in enumerate(i2r)}

    # Write triples to CSV
    print('Writing integer triples.')
    triples, c = doc.search_triples('', '', '')
    with gzip.open('triples.int.csv.gz', 'wt') as file:

        for s, p, o in tqdm(triples, total=c):

            sp = kg.entity_hdt(s)
            op = kg.entity_hdt(o)

            file.write(f'{e2i[sp]}, {r2i[p]}, {e2i[op]}\n')

    return (e2i, r2i)

def generate_csv_splits(splits, e2i, r2i):
    if len(splits) <= 0:
        return

    train_path, test_path, valid_path = splits[:3]
    # Load test/train/valid/meta
    g_train = Graph()
    with gzip.open(train_path, 'rb') as gzf:
        g_train.parse(gzf, format='nt')
    g_test = Graph()
    with gzip.open(test_path, 'rb') as gzf:
        g_test.parse(gzf, format='nt')
    g_valid = Graph()
    with gzip.open(valid_path, 'rb') as gzf:
        g_valid.parse(gzf, format='nt')

    g_meta = Graph()
    if len(splits) == 4:
        meta_path = splits[3]
        with gzip.open(meta_path, 'rb') as gzf:
            g_meta.parse(gzf, format='nt')

    c2i = dict()
    i = 0
    for g in [g_train, g_test, g_valid, g_meta]:
        classes = set(g.objects())
        for c in classes:
            if c not in c2i.keys():
                c2i[c] = i
                i += 1

    with open('all.int.csv', 'w') as allfile:
        allwriter = csv.writer(allfile, delimiter=',')
        allwriter.writerow(['index', 'class'])

        with open('training.int.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['index', 'class'])
            for s, p, o in g_train.triples((None, None, None)):
                s_idx = e2i[(str(s), 'iri')]
                o_idx = c2i[o]

                writer.writerow([s_idx, o_idx])
                allwriter.writerow([s_idx, o_idx])

        with open('testing.int.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['index', 'class'])
            for s, p, o in g_test.triples((None, None, None)):
                s_idx = e2i[(str(s), 'iri')]
                o_idx = c2i[o]

                writer.writerow([s_idx, o_idx])
                allwriter.writerow([s_idx, o_idx])

        with open('validation.int.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['index', 'class'])
            for s, p, o in g_valid.triples((None, None, None)):
                s_idx = e2i[(str(s), 'iri')]
                o_idx = c2i[o]

                writer.writerow([s_idx, o_idx])
                allwriter.writerow([s_idx, o_idx])

        if len(g_meta) > 0:
            with open('meta-testing.int.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['index', 'class'])
                for s, p, o in g_meta.triples((None, None, None)):
                    s_idx = e2i[(str(s), 'iri')]
                    o_idx = c2i[o]

                    writer.writerow([s_idx, o_idx])
                    allwriter.writerow([s_idx, o_idx])

def generate_csv(doc, splits):
    e2i, r2i = generate_csv_context(doc)
    generate_csv_splits(splits, e2i, r2i)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1 or len(args) > 5:
        print("USAGE: ./hdt2csv.py <graph_stripped.hdt> [<train_set.nt.gz> <test_set.nt.gz> <valid_set.nt.gz> [<meta_set.nt.gz>]]")

    hdtfile = args[0]
    doc = hdt.HDTDocument(hdtfile)

    generate_csv(doc, args[1:])

