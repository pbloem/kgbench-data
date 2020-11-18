from .util import here, tic, toc
import numpy as np
import os
from os.path import join as j
import pandas as pd
import gzip

import torch

class Data:

    triples = None

    i2r, r2i = None, None
    i2e, e2i = None, None


    num_entities = None
    """ Total number of relations in the graph """

    num_relations = None
    """ Total number of entities in the graph """

    num_classes = None
    """ Total number of classes in the classification task """

    training = None
    """ Training data: a matrix with entity indices in column 0 and class indices in column 1.
        In non-final mode, this is the training part of the train/val/test split. In final mode, this is 
        the training part and the validation part.  
    """
    withheld = None
    """ Validation/testing data: a matrix with entity indices in column 0 and class indices in column 1.
        In non-final mode this is the validation data. In final mode this is the testing data.
    """

    def __init__(self, dir, final=False, use_torch=False):
        if dir is not None:
            self.torch = use_torch

            tic()
            self.triples = fastload(j(dir, 'triples.int.csv.gz'))
            print(f'loaded triples ({toc():.4}s).')

            # tic()
            # with gzip.open(j(dir, 'triples.int.csv.gz')) as input:
            #     self.triples = np.loadtxt(input, dtype=np.int, delimiter=',')
            # print(f'loaded triples ({toc():.4}s).')
            #
            # print(ftriples.shape, self.triples.shape)
            # assert (ftriples != self.triples).sum() == 0

            self.i2r, self.r2i = load_indices(j(dir, 'relations.int.csv'))
            self.i2e, self.e2i = load_indices(j(dir, 'entities.int.csv'))

            self.num_entities  = len(self.i2e)
            self.num_relations = len(self.i2r)

            train, val, test = \
                np.loadtxt(j(dir, 'training.int.csv'),   dtype=np.int, delimiter=','), \
                np.loadtxt(j(dir, 'validation.int.csv'), dtype=np.int, delimiter=','), \
                np.loadtxt(j(dir, 'testing.int.csv'),    dtype=np.int, delimiter=',')

            if final:
                self.training = np.concatenate([train, val], axis=0)
                self.withheld = test
            else:
                self.training = train
                self.withheld = val

            self.final = final

            self.num_classes = len(set(self.training[:, 1]))

            print('loaded data.')

            print(f'   {len(self.triples)} triples')

            if use_torch: # this should be constant-time/memory
                self.triples = torch.from_numpy(self.triples)
                self.training = torch.from_numpy(self.training)
                self.withheld = torch.from_numpy(self.withheld)


def load(name, final=False, torch=False, prune_dist=None):
    """
    Returns the requested dataset.
    :param name: One of amfull, am1k, wd-people
    :param final: Loads the test/train split instead of the validation train split. In this case the training data
    consists of both training and validation.
    :return: A pair (triples, meta). `triples` is a numpy 2d array of datatype uint32 contianing integer-encoded
    triples. `meta` is an object of metadata containing the following fields:
     * e: The number of entities
     * r: The number of relations
     * i2r:
    """

    if name == 'amfull':
        data = Data(here('../datasets/amfull'), final=final, use_torch=torch)
    elif name == 'am1k':
        data =  Data(here('../datasets/am1k'), final=final, use_torch=torch)
    else:
        raise Exception(f'Dataset {name} not recognized.')

    if prune is not None:
        tic()
        data = prune(data, n=prune_dist)
        print(f'pruned ({toc():.4}s).')

    return data

def load_indices(file):

    df = pd.read_csv(file)

    i2l = df['label'].tolist()
    l2i = {l:i for i, l in enumerate(i2l)}

    return i2l, l2i

def prune(data : Data, n=2):
    """
    Prune a given dataset. That is reduce the number of triples to an n-hop neighborhood around the labeled nodes. This
    can save a lot of memory if the model being used is known to look only to a certain depth in the graph.

    Note that switching between non-final and final mode will result in different pruned graphs

    :param data:
    :return:
    """

    data_triples = data.triples
    data_training = data.training
    data_withheld = data.withheld

    if data.torch:
        data_triples = data_triples.numpy()
        data_training = data_training.numpy()
        data_withheld = data_withheld.numpy()

    assert n >= 1

    entities = set()

    for e in data_training[:, 0]:
        entities.add(e)
    for e in data_withheld[:, 0]:
        entities.add(e)

    entities_add = set()
    for _ in range(n):
        for s, p, o in data_triples:
            if s in entities:
                entities_add.add(o)
            if o in entities:
                entities_add.add(s)
        entities.update(entities_add)

    # new index to old index
    n2o = list(entities)
    o2n = {o: n for n, o in enumerate(entities)}

    nw = Data(dir=None)

    nw.num_entities = len(n2o)
    nw.num_relations = data.num_relations

    nw.i2e = [data.i2e[n2o[i]] for i in range(len(n2o))]
    nw.e2i = {e: i for i, e in enumerate(nw.i2e)}

    nw.i2r = data.i2r
    nw.r2i = data.r2i

    # count the new number of triples
    num = 0
    for s, p, o in data_triples:
        if s in entities and o in entities:
            num += 1

    nw.triples = np.zeros((num, 3), dtype=int)

    row = 0
    for s, p, o in data_triples:
        if s in entities and o in entities:
            s, o =  o2n[s], o2n[o]
            nw.triples[row, :] = (s, p, o)
            row += 1

    nw.training = data_training.copy()
    for i in range(nw.training.shape[0]):
        nw.training[i, 0] = o2n[nw.training[i, 0]]

    nw.withheld = data_withheld.copy()
    for i in range(nw.withheld.shape[0]):
        nw.withheld[i, 0] = o2n[nw.withheld[i, 0]]

    nw.num_classes = data.num_classes

    nw.final = data.final
    nw.torch = data.torch
    if nw.torch:  # this should be constant-time/memory
        nw.triples = torch.from_numpy(nw.triples)
        nw.training = torch.from_numpy(nw.training)
        nw.withheld = torch.from_numpy(nw.withheld)

    return nw

def fastload(file):
    """
    Quickly (?) load a matrix of triples
    :param input:
    :return:
    """
    with gzip.open(file, 'rt') as input:
        lines = 0
        for _ in input:
            lines += 1

    result = np.zeros((lines, 3), dtype=np.int)

    with gzip.open(file, 'rt') as input:
        for i, line in enumerate(input):
            s, p, o = str(line).split(',')
            s, p, o = int(s), int(p), int(o)
            result[i, :] = (s, p, o)

    return result