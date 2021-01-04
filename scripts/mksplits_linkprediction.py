#!/usr/bin/env python

from math import ceil
import gzip
import sys

import numpy as np
import pandas as pd


NUM_SAMPLES_TEST = 20000
NUM_SAMPLES_VALID = 20000
NUM_SAMPLES_META = 20000

def divide_members(members, test_ratio, valid_ratio, meta_ratio):
    num_members = len(members)

    if num_members < 4:
        return (members, [], [], [])

    test_portion = ceil(num_members * test_ratio)
    valid_portion = ceil(num_members * valid_ratio)
    meta_portion = ceil(num_members * meta_ratio)
    train_portion = num_members - test_portion - valid_portion - meta_portion

    assert train_portion > test_portion

    return (members[:train_portion],
            members[train_portion:train_portion+test_portion],
            members[train_portion+test_portion:num_members-meta_portion],
            members[train_portion+test_portion+valid_portion:])

def generate_splits(triples, split_sizes, stratified=True):
    if len(split_sizes) < 2:
        print("USAGE: ./mksplits_linkprediction.py <triples.int.nt.gz> [<test_set_size> <valid_set_size> [<meta_set_size>]]")
        sys.exit(1)

    num_samples = len(triples)
    num_samples_test, num_samples_valid = split_sizes[:2]

    test_ratio = num_samples_test / num_samples
    valid_ratio = num_samples_valid / num_samples

    meta_ratio = 0
    if len(split_sizes) > 2:
        num_samples_meta = split_sizes[2]
        meta_ratio = num_samples_meta / num_samples

    samples_map = dict()
    for i, (_, p, _) in enumerate(triples):
        if p not in samples_map.keys():
            samples_map[p] = list()
        samples_map[p].append(i)

    train_set, test_set, valid_set, meta_set = list(), list(), list(), list()
    if not stratified:
        samples = list()
        for members in samples_map.values():
            samples.extend(members)

        train, test, valid, meta = divide_members(members, test_ratio,
                                                  valid_ratio, meta_ratio)

        for sample in train:
            train_set.append(sample)
        for sample in test:
            test_set.append(sample)
        for sample in valid:
            valid_set.append(sample)
        for sample in meta:
            meta_set.append(sample)
    else:
        for members in samples_map.values():
            train, test, valid, meta = divide_members(members, test_ratio,
                                                     valid_ratio, meta_ratio)

            for sample in train:
                train_set.append(sample)
            for sample in test:
                test_set.append(sample)
            for sample in valid:
                valid_set.append(sample)
            for sample in meta:
                meta_set.append(sample)

    print('split distribution:')
    print('- %d train' % len(train_set))
    print('- %d test' % len(test_set))
    print('- %d valid' % len(valid_set))
    if len(meta_set) > 0:
        print('- %d meta' % len(meta_set))

    out = np.zeros((num_samples, 2), dtype=np.int)  # train = 0
    out[:, 0] = np.arange(num_samples)
    out[test_set, 1] = 1
    out[valid_set, 1] = 2
    if len(meta_set) > 0:
        out[meta_set, 1] = 3

    return pd.DataFrame(out, columns=["index", "split"])

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1 or len(args) > 4:
        print("USAGE: ./mksplits_linkprediction.py <triples.int.nt.gz> [<test_set_size> <valid_set_size> [<meta_set_size>]]")
        sys.exit(1)

    graph_path = args[0]
    splits = np.empty(0)
    with gzip.open(graph_path) as gzf:
        triples = pd.read_csv(gzf, delimiter=',').to_numpy()
        splits = generate_splits(triples, [int(v) for v in args[1:]], stratified=True)

    splits.to_csv('./linkprediction_splits.csv', index=False)
