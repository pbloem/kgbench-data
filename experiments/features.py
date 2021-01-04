"""
Run the RGCN baseline of the given dataset
"""

import fire, sys, tqdm

from kgbench import load, tic, toc, d
import numpy as np
from sklearn.linear_model import LogisticRegression

"""
A simple feature-based baseline.

Computes the k most distinctive features on the training set, by information gain, and trains a linear classifier on 
these. For a given instance node i, a feature is a binary value indicating the presence amopng adjacent edges of either:
- a particular relation (in or outgoing)
- a particular directed relation
- a directed relation with a particular neighbor node.

A linear logistic regression model is trained on this data. 

  
"""

MINFREQ = 3

from collections import Counter

def put(dict, key, c, nc):
    if not key in dict:
        dict[key] = [0] * nc

    dict[key][c] += 1

def log2(arr):
    return np.log2(arr, where=arr != 0.0)

def tostr(feat, data):
    if len(feat) == 1:
        return data.i2r[feat[0]]
    if len(feat) == 2:
        return  ('<-' if feat[1] else '->') + data.i2r[feat[0]]
    if len(feat) == 3:
        return  ('<-' if feat[1] else '->') + data.i2r[feat[0]] + ', ' + data.i2e[feat[2]]

    raise Exception(str(feat) + '?')

def has_feature(inst, feat, rels, inrels, outrels, infull, outfull):
    if len(feat) == 1:
        return feat[0] in rels[inst]

    if len(feat) == 2:
        dict = inrels if feat[1] else outrels
        return feat[0] in dict[inst] if inst in dict else False

    if len(feat) == 3:
        dict = infull if feat[1] else outfull
        return (feat[0], feat[2]) in dict[inst] if inst in dict else False


def go(name='mdgenre', final=True, numfeatures=2_000, printweights=False):

    print('arguments: ', ' '.join([f'{k}={v}' for k, v in locals().items()]))

    data = load(name, torch=False, prune_dist=1, final=final)

    print(f'{data.triples.shape[0]} triples')
    print(f'{data.num_entities} entities')
    print(f'{data.num_relations} relations')

    print(f'{data.training.shape[0]} training instances')

    tic()

    nc = data.num_classes

    tallies = {}
    # -- maps potential features to tallies per class

    rels = {}
    inrels, outrels = {}, {}
    infull, outfull = {}, {}

    # Create some dictionaries for easy access
    print('Creating dicts.')
    for s, p, o in tqdm.tqdm(data.triples):

        if s not in rels:
            rels[s] = set()
        if o not in rels:
            rels[o] = set()
        rels[s].add(p)
        rels[o].add(p)

        if o not in inrels:
            inrels[o] = set()
        if s not in outrels:
            outrels[s] = set()

        outrels[s].add(p)
        inrels[o].add(p)

        if o not in infull:
            infull[o] = set()
        if s not in outfull:
            outfull[s] = set()

        infull[o].add((p, s))
        outfull[s].add((p, o))

    # Compute the features
    # - Tallies for all features
    print('Tallying features.')
    for i in tqdm.trange(data.training.shape[0]):

        inst, cls = data.training[i, :]

        for r in rels[inst] if inst in rels else []:
            put(tallies, (r,), cls, nc)

        # relation present in a particular direction (p, bool)
        for r in inrels[inst] if inst in inrels else []:
            put(tallies, (r, True), cls, nc)
        for r in outrels[inst] if inst in outrels else []:
            put(tallies, (r, False), cls, nc)

        # relation present in a particular direction, with a particular neighbor (p, bool, neighbor)
        for rel, ngh in infull[inst] if inst in infull else []:
            put(tallies, (rel, True, ngh), cls, nc)
        for rel, ngh in outfull[inst] if inst in outfull else []:
            put(tallies, (rel, False, ngh), cls, nc)

    # - Compute information gain for all features
    # -- The gain is computed as for a decision stub where we split on the feature being present or not.
    # global class tally
    gc = [0] * nc
    for cls in data.training[:, 1]:
        gc[cls] += 1

    gc = np.asarray(gc)
    gcp = gc / gc.sum(keepdims=True)         # relative frequencies
    gce = - (gcp * log2(gcp)).sum()   # entropy before split

    gains = Counter()
    for feat, tl in tallies.items():

        # class tallies for the feature being present and absent
        present = np.asarray(tl)
        absent = gc - present

        psum, asum = present.sum(), absent.sum()
        if psum <= MINFREQ or asum <= MINFREQ:
            continue

        # relative frequencies and priors
        presentp, absentp = present / present.sum(keepdims=True), absent / absent.sum(keepdims=True)

        pprior, aprior = psum / (asum + psum), asum / (asum + psum)

        # entropies
        presente, absente = - (presentp * log2(presentp)).sum(), - (absentp * log2(absentp)).sum()

        gains[feat] = gce - (pprior * presente + aprior * absente)

    # for feat, gain in gains.most_common(2000):
    #
    #     print(f'{gain:.4}\t{tostr(feat, data)}')

    i2f = [feat for feat, gain in gains.most_common(numfeatures)]
    f2i = {f:i for i, f in enumerate(i2f)}

    instances = []
    classes = data.training[:, 1]

    for inst in data.training[:, 0]:
        instance = []
        for feat in i2f:
            instance.append(has_feature(inst, feat, rels, inrels, outrels, infull, outfull))

        instances.append(instance)

    instances = np.asarray(instances)

    instances_wh = []
    classes_wh = data.withheld[:, 1]

    for inst in data.withheld[:, 0]:
        instance = []
        for feat in i2f:
            instance.append(has_feature(inst, feat, rels, inrels, outrels, infull, outfull))

        instances_wh.append(instance)

    instances_wh = np.asarray(instances_wh)

    print('Fitting model.')
    lr = LogisticRegression(multi_class='multinomial', max_iter=10_00)
    lr.fit(instances, classes)


    print('Model fitted.')
    print(f'   training acc {lr.score(instances, classes)}')
    print(f'   withheld acc {lr.score(instances_wh, classes_wh)}')

    if printweights:

        print(lr.coef_.shape) # classes x features
        pairs = [(c, feat) for c, feat in zip(lr.coef_.T, i2f)]

        pairs.sort(key=lambda p : - np.linalg.norm(p[0]))

        for c, f in pairs:
            print(np.linalg.norm(c), f, data.i2r[f[0]], data.i2e[f[2]] if len(f) > 2 else '')



if __name__ == '__main__':

    print('arguments ', ' '.join(sys.argv))
    fire.Fire(go)