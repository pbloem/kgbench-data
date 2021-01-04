#!/usr/bin/env python

from math import ceil
import gzip

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
            members[-meta_portion:])

def main(triples_csv, stratified=True):
    triples = pd.read_csv(triples_csv, delimiter=',').to_numpy()

    num_samples = len(triples)
    test_ratio = NUM_SAMPLES_TEST / num_samples
    valid_ratio = NUM_SAMPLES_VALID / num_samples
    meta_ratio = NUM_SAMPLES_META / num_samples

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
    print('- %d meta' % len(meta_set))

    out = np.zeros((num_samples, 2), dtype=np.int)  # train = 0
    out[:, 0] = np.arange(num_samples)
    out[test_set, 1] = 1
    out[valid_set, 1] = 2
    out[meta_set, 1] = 3

    return pd.DataFrame(out, columns=["index", "split"])

if __name__ == "__main__":
    splits = np.empty(0)
    with gzip.open('./triples.int.csv.gz') as gzf:
        splits = main(gzf, stratified=True)

    splits.to_csv('./linkprediction_splits.csv', index=False)
