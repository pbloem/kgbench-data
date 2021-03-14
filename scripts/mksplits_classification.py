#!/usr/bin/env python

import gzip
from math import ceil
import sys

from rdflib import Graph


def divide_members(members, test_ratio, valid_ratio, meta_ratio):
    num_members = len(members)

    test_portion = ceil(num_members * test_ratio)
    valid_portion = ceil(num_members * valid_ratio)
    meta_portion = ceil(num_members * meta_ratio)
    train_portion = num_members - test_portion - valid_portion - meta_portion

    return (members[:train_portion],
            members[train_portion:train_portion+test_portion],
            members[train_portion+test_portion:num_members-meta_portion],
            members[train_portion+test_portion+valid_portion:])

def generate_splits(g, split_sizes, stratified):
    if len(split_sizes) < 2:
        print("USAGE: ./mksplits_classification.py <samples.nt.gz> [<test_set_size> <valid_set_size> [<meta_set_size>]]")
        sys.exit(1)

    num_samples = 0
    samples_map = dict()
    for s,p,o in g.triples((None, None, None)):
        if o not in samples_map.keys():
            samples_map[o] = list()
        samples_map[o].append((s,p,o))
        num_samples += 1

    num_samples_test, num_samples_valid = split_sizes[:2]
    test_ratio = num_samples_test / num_samples
    valid_ratio = num_samples_valid / num_samples

    meta_ratio = 0
    if len(split_sizes) > 2:
        num_samples_meta = split_sizes[2]
        meta_ratio = num_samples_meta / num_samples

    for sample_class, members in samples_map.items():
        samples_map[sample_class] = sorted(members)  # ensure reproducability

    train_set, test_set, valid_set, meta_set = Graph(), Graph(), Graph(), Graph()
    if not stratified:
        samples = list()
        for members in samples_map.values():
            samples.extend(members)

        train, test, valid, meta = divide_members(members, test_ratio,
                                                  valid_ratio, meta_ratio)

        for sample in train:
            train_set.add(sample)
        for sample in test:
            test_set.add(sample)
        for sample in valid:
            valid_set.add(sample)
        for sample in meta:
            meta_set.add(sample)
    else:
        for members in samples_map.values():
            train, test, valid, meta = divide_members(members, test_ratio,
                                                     valid_ratio, meta_ratio)

            for sample in train:
                train_set.add(sample)
            for sample in test:
                test_set.add(sample)
            for sample in valid:
                valid_set.add(sample)
            for sample in meta:
                meta_set.add(sample)

    return (train_set, test_set, valid_set, meta_set)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1 or len(args) > 4:
        print("USAGE: ./mksplits_classification.py <samples.nt.gz> [<test_set_size> <valid_set_size> [<meta_set_size>]]")
        sys.exit(1)

    graph_path = args[0]
    g = Graph()
    with gzip.open(graph_path, 'rb') as f:
        g.parse(data=f.read(), format='nt')

    train_set, test_set, valid_set, meta_set = generate_splits(g, [int(v) for v in args[1:]], stratified=True)


    n = 0
    print("Split Distribution:")
    for name, graph in zip(('train', 'test', 'valid', 'meta'),
                           (train_set, test_set, valid_set, meta_set)):
        if len(graph) <= 0:
            continue

        n += len(graph)
        print(" - %s : %d" % (name, len(graph)))

    print("- total: %d" % n)

    for filename, graph in zip(('train_set.nt.gz',
                                'test_set.nt.gz',
                                'valid_set.nt.gz',
                                'meta_set.nt.gz'),
                               (train_set, test_set, valid_set, meta_set)):
        if len(graph) <= 0:
            continue

        with gzip.open('./'+filename, 'wb') as f:
            f.write(graph.serialize(format='nt'))
