#!/usr/bin/env python

import gzip
from math import ceil

from rdflib import Graph,URIRef


CLASS_PRED = URIRef('https://data.labs.pdok.nl/rce/def/monumentCode')

def divide_members(members, train_test_ratio, train_valid_ratio):
    num_members = len(members)

    test_portion = ceil(num_members * train_test_ratio[1])
    remain = num_members - test_portion
    assert remain > test_portion

    valid_portion = ceil(remain * train_valid_ratio[1])
    train_portion = remain - valid_portion
    assert train_portion > (valid_portion + test_portion)

    return (members[:train_portion],
            members[train_portion:train_portion+test_portion],
            members[-valid_portion:])

def adjust_ratio(num_samples, train_test_ratio, train_valid_ratio,
                 test_samples_min, valid_samples_min):
    test_portion = ceil(num_samples * train_test_ratio[1])
    remain = num_samples - test_portion
    valid_portion = ceil(remain * train_valid_ratio[1])
    train_portion = remain - valid_portion

    if (test_portion < test_samples_min or valid_portion < valid_samples_min)\
       and train_portion > (test_portion+valid_portion):
        test_portion = test_samples_min / num_samples
        train_test_ratio = (1.0 - test_portion, test_portion)

        remain = num_samples - ceil(num_samples * test_portion)
        valid_portion = valid_samples_min / remain
        train_valid_ratio = (1.0 - valid_portion, valid_portion)

    return (train_test_ratio, train_valid_ratio)

def create_splits(g, train_test_ratio, train_valid_ratio, test_samples_min,
                  valid_samples_min, stratified):
    num_samples = 0
    samples_map = dict()
    for s,p,o in g.triples((None, CLASS_PRED, None)):
        if o not in samples_map.keys():
            samples_map[o] = list()
        samples_map[o].append((s,p,o))
        num_samples += 1

    train_test_ratio, train_valid_ratio = adjust_ratio(num_samples,
                                                       train_test_ratio,
                                                       train_valid_ratio,
                                                       test_samples_min,
                                                       valid_samples_min)

    for sample_class, members in samples_map.items():
        samples_map[sample_class] = sorted(members)  # ensure reproducability

    train_set, test_set, valid_set = Graph(), Graph(), Graph()
    if not stratified:
        samples = list()
        for members in samples_map.values():
            samples.extend(members)

        train, test, valid = divide_members(members,
                                            train_test_ratio,
                                            train_valid_ratio)

        for sample in train:
            train_set.add(sample)
        for sample in test:
            test_set.add(sample)
        for sample in valid:
            valid_set.add(sample)
    else:
        for members in samples_map.values():
            train, test, valid = divide_members(members,
                                                train_test_ratio,
                                                train_valid_ratio)

            for sample in train:
                train_set.add(sample)
            for sample in test:
                test_set.add(sample)
            for sample in valid:
                valid_set.add(sample)

    return (train_set, test_set, valid_set)

def main(file_path='./', train_test_ratio=(.9, .1), train_valid_ratio=(.9, .1),
         test_samples_min=10000, valid_samples_min=5000, stratified=True):

    g = Graph()
    with gzip.open(file_path+'targets.nt.gz', 'rb') as f:
        g.parse(data=f.read(), format='nt')

    train_set, test_set, valid_set = create_splits(g, train_test_ratio,
                                                   train_valid_ratio,
                                                   test_samples_min,
                                                   valid_samples_min,
                                                   stratified)

    return (train_set, test_set, valid_set)

if __name__ == "__main__":
    train_set, test_set, valid_set = main('./')

    print("Split Distribution:")
    for name, graph in zip(('train', 'test', 'valid'),
                           (train_set, test_set, valid_set)):
        print(" - %s : %d" % (name, len(graph)))

    for filename, graph in zip(('train_set.nt.gz',
                                'test_set.nt.gz',
                                'valid_set.nt.gz'),
                               (train_set, test_set, valid_set)):
        with gzip.open('./'+filename, 'wb') as f:
            f.write(graph.serialize(format='nt'))
