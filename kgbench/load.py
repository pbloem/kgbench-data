from .util import here, tic, toc
import numpy as np
import os
from os.path import join as j
import pandas as pd
import gzip, base64, io, sys

from skimage import io as skio

import torch

I2D = [
    'none',
    'blank_node',
    'uri',
    'http://www.w3.org/2001/XMLSchema#b64string',
    'http://www.w3.org/2001/XMLSchema#date',
    'http://www.w3.org/2001/XMLSchema#decimal',
    'http://www.w3.org/2001/XMLSchema#positiveInteger'
]

""" List of string identifies for all datatypes used in the various datasets."""

D2I = {d: i for i, d in enumerate(I2D)}
""" Dictionary mapping datatypes to integer indices."""

class Data:

    triples = None
    """ The edges of the knowledge graph (the triples), represented by their integer indices. A (m, 3) numpy 
        or pytorch array.
    """

    i2r, r2i = None, None

    i2e = None
    """ A mapping from an integer index to an entity representation. An entity is either a simple string indicating the label 
        of the entity (a url, blank node or literal), or it is a pair indicating the datatype and the label (in that order).
    """

    e2i = None
    """ A dictionary providing the inverse mappring of i2e
    """

    num_entities = None
    """ Total number of distinct entities (nodes) in the graph """

    num_relations = None
    """ Total number of distinct relation types in the graph """

    num_classes = None
    """ Total number of classes in the classification task """

    training = None
    """ Training data: a matrix with entity indices in column 0 and class indices in column 1.
        In non-final mode, this is the training part of the train/val/test split. In final mode, the training part, 
        possibly concatenated with the validation data.
    """

    withheld = None
    """ Validation/testing data: a matrix with entity indices in column 0 and class indices in column 1.
        In non-final mode this is the validation data. In final mode this is the testing data.
    """

    def __init__(self, dir, final=False, use_torch=False, catval=False):
        if dir is not None:
            self.torch = use_torch

            tic()
            self.triples = fastload(j(dir, 'triples.int.csv.gz'))
            print(f'loaded triples ({toc():.4}s).')

            self.i2r, self.r2i = load_indices(j(dir, 'relations.int.csv'))
            self.i2e, self.e2i = load_entities(j(dir, 'entities.int.csv'))

            self.num_entities  = len(self.i2e)
            self.num_relations = len(self.i2r)

            train, val, test = \
                np.loadtxt(j(dir, 'training.int.csv'),   dtype=np.int, delimiter=','), \
                np.loadtxt(j(dir, 'validation.int.csv'), dtype=np.int, delimiter=','), \
                np.loadtxt(j(dir, 'testing.int.csv'),    dtype=np.int, delimiter=',')

            if final and catval:
                self.training = np.concatenate([train, val], axis=0)
                self.withheld = test
            elif final:
                self.training = train
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


    def get_pil_images(self, dtype='http://www.w3.org/2001/XMLSchema#b64string'):
        """
        Retrieves the entities with the given datatype as images. This assumes that the labels are b64 encoded images.

        :param dtype:
        :return: A list of PIL image objects
        """
        from PIL import Image

        res = []
        # Take in base64 string and return cv image
        num_noparse = 0
        for b64 in self.get_strings(dtype):
            imgdata = base64.b64decode(b64, '-_')
            try:
                res.append(Image.open(io.BytesIO(imgdata)))
            except:
                num_noparse += 1
                res.append(Image.new('RGB', (1, 1)))
                # -- if the image can't be parsed, we insert a 1x1 black image
                # TODO 214 items in AMplus are error messages

        # print(num_noparse, 'unparseable', len([r for r in res if r is not None]), 'parseable')

        return res

    # def get_image_tensor(self, dtype='http://www.w3.org/2001/XMLSchema#b64string'):
    #     """
    #     Retrieves the entities with the given datatype as images. This assumes that the labels are b64 encoded images.
    #
    #     :param dtype:
    #     :return: A 4-tensor of shape (n, c, h, w) containing all entities of the given datatype.
    #     """
    #
    #     res = []
    #     # Take in base64 string and return cv image
    #     for b64 in self.get_strings(dtype):
    #         imgdata = base64.b64decode(b64, '-_')
    #         try:
    #             h, w, c = skio.imread(io.BytesIO(imgdata)).shape
    #             break
    #         except:
    #             continue
    #
    #     for b64 in self.get_strings(dtype):
    #         imgdata = base64.b64decode(b64, '-_')
    #         try:
    #             image = skio.imread(io.BytesIO(imgdata)).transpose(0, 2)[None, :, :, :]
    #             assert image.shape == (1, c, h, w), f'All images expected to have same dimensions {(c, h, w)}'
    #             res.append(image)
    #         except:
    #             res.append(np.zeros((1, c, h, w)))
    #             # TODO add zeros for unparseable images for now.
    #
    #     res = np.concatenate(res, axis=0)
    #
    #     return torch.from_numpy(res) if self.torch else res

    _dt_i2e = {}
    _dt_e2i = {}
    def get_strings(self, dtype, copy=True):
        """
        Retrieves all entities of the given datatype.

        :param copy: If False, returns the list which back this data object. Faster, but list should not be modified.
        :return:
        """
        if dtype not in self._dt_i2e:
            self._dt_i2e[dtype] = [label for i, (label, dt) in self.i2e if dt == dtype]
            self._dt_e2i[dtype] = {label: i for i, label in enumerate(self._dt_i2e[dtype])}

        return list(self._dt_i2e[dtype]) if copy else self._dt_i2e[dtype]

    def get_dtidx(self, dtype, copy=True):
        """
        Returns a dictionary mapping an entity label to its index among other entities of the same datatype.

        That is, and entity "0.3"^^<http://www.w3.org/2001/XMLSchema#decimall> May be 10345 among all entities, but 678
        among only the entities with this particular datatype.

        :param entity_idx:
        :param dtype:
        :return:
        """
        if dtype not in self._dt_maps:
            self.get_strings(dtype)

        return dict(self._dt_e2i[dtype]) if copy else self._dt_e2i[dtype]

    def dt_index(self, enidx):
        """
        Converts a global index of an entity to its index among entities with the same datatype.

        :param enidx:
        :return:
        """
        dt, lbl = self.i2e[enidx]

        return self.get_dtidx(dt, copy=False)[lbl]

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

    if name == 'micro':
        return micro(final, torch)
        # -- a miniature dataset for unit testing

    if name in ['aifb', 'am1k', 'amfull', 'amplus']:
        data = Data(here(f'../datasets/{name}'), final=final, use_torch=torch)
    else:
        raise Exception(f'Dataset {name} not recognized.')

    if prune_dist is not None:
        tic()
        data = prune(data, n=prune_dist)
        print(f'pruned ({toc():.4}s).')

    return data

def micro(final=True, use_torch=False):
    """
    Micro dataset for unit testing.

    :return:
    """

    data = Data(None)

    data.num_entities = 5
    data.num_relations = 2
    data.num_classes = 2

    data.i2e = [str(i) for i in range(data.num_entities)]
    data.i2r = [str(i) for i in range(data.num_entities)]

    data.e2i = {i:e for e, i in enumerate(data.i2e)}
    data.r2i = {i:r for r, i in enumerate(data.i2e)}

    data.final = final
    data.triples = np.asarray(
        [[0, 0, 1], [1, 0, 2], [0, 0, 2], [2, 1, 3], [4, 1, 3], [4, 1, 0] ],
        dtype=np.int
    )

    data.training = np.asarray(
        [[1, 0], [2, 0]],
        dtype=np.int
    )

    data.withheld = np.asarray(
        [[3, 1], [3, 1]],
        dtype=np.int
    )

    data.torch = use_torch
    if torch: # this should be constant-time/memory
        data.triples  = torch.from_numpy(data.triples)
        data.training = torch.from_numpy(data.training)
        data.withheld = torch.from_numpy(data.withheld)

    return data

def load_indices(file):

    df = pd.read_csv(file)

    idxs = df['index'].tolist()
    labels = df['label'].tolist()

    i2l = list(zip(idxs, labels))
    i2l.sort(key=lambda p: p[0])

    l2i = {l:i for i, l in i2l}

    return i2l, l2i

def load_entities(file):

    df = pd.read_csv(file)

    assert len(df.columns) == 3, 'Entity file should have three columns (index, datatype and label)'

    idxs = df['index']      .tolist()
    dtypes = df['datatype'] .tolist()
    labels = df['label']    .tolist()

    ents = zip(labels, dtypes)

    i2e = list(zip(idxs, ents))
    i2e.sort(key=lambda p: p[0])

    e2i = {e: i for i, e in i2e}

    return i2e, e2i

def prune(data : Data, n=2):
    """
    Prune a given dataset. That is, reduce the number of triples to an n-hop neighborhood around the labeled nodes. This
    can save a lot of memory if the model being used is known to look only to a certain depth in the graph.

    Note that switching between non-final and final mode will result in different pruned graphs.

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
    Quickly load an (m, 3) matrix of integer triples
    :param input:
    :return:
    """
    # count the number of lines
    with gzip.open(file, 'rt') as input:
        lines = 0
        for _ in input:
            lines += 1

    # prepare a zero metrix
    result = np.zeros((lines, 3), dtype=np.int)

    # fill the zero matrix with the values from the file
    with gzip.open(file, 'rt') as input:
        for i, line in enumerate(input):
            s, p, o = str(line).split(',')
            s, p, o = int(s), int(p), int(o)
            result[i, :] = (s, p, o)

    return result